#include "numerics/hdg/euler/smooth_euler.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using dg::hdg::euler::ConservedState;
using dg::hdg::euler::ByNodesIndex;
using dg::hdg::euler::ComponentFromByNodesIndex;
using dg::hdg::euler::DecodeSignedVDof;
using dg::hdg::euler::FluxJacobian;
using dg::hdg::euler::FluxVector;
using dg::hdg::euler::ManufacturedConservative;
using dg::hdg::euler::ManufacturedMaximumWaveSpeed;
using dg::hdg::euler::ManufacturedSource;
using dg::hdg::euler::NormalFlux;
using dg::hdg::euler::NormalFluxJacobian;
using dg::hdg::euler::PhysicsStatus;
using dg::hdg::euler::RunSmoothEulerExperiment;
using dg::hdg::euler::ScalarFromByNodesIndex;
using dg::hdg::euler::SmoothEulerExperiment;
using dg::hdg::euler::ExperimentEvidence;
using dg::hdg::euler::ExperimentStatus;
using dg::hdg::euler::TrialRejectionReason;
using dg::hdg::euler::SolvedMeshEvidence;
using dg::hdg::euler::ConvergenceEvidence;
using dg::hdg::euler::ErrorIntegralEvidence;
using dg::hdg::euler::LocalAssemblyAuditSnapshot;
using dg::hdg::euler::LinearAuditSnapshot;
using dg::hdg::euler::TrialAuditSnapshot;
using dg::hdg::euler::LocalLinearizationSnapshot;
using dg::hdg::euler::backward_euler_dt;
using dg::hdg::euler::dimension;
using dg::hdg::euler::fixed_scalar_tau;
using dg::hdg::euler::conserved_components;
using dg::hdg::euler::gamma;
using dg::hdg::euler::experiment_epsilon;

bool Close(const double left, const double right, const double tolerance)
{
  return std::isfinite(left) && std::isfinite(right) &&
         std::isfinite(tolerance) && tolerance >= 0.0 &&
         std::abs(left - right) <=
             tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

bool FiniteVector(const std::vector<double>& values)
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return std::isfinite(value); });
}

double InfinityNorm(const std::vector<double>& values)
{
  if (!FiniteVector(values)) return std::numeric_limits<double>::infinity();
  double result = 0.0;
  for (const double value : values) { result = std::max(result, std::abs(value)); }
  return result;
}

double DifferenceNorm(const std::vector<double>& left,
                      const std::vector<double>& right)
{
  if (left.size() != right.size() || !FiniteVector(left) || !FiniteVector(right))
  {
    return std::numeric_limits<double>::infinity();
  }
  double result = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i)
  {
    result = std::max(result, std::abs(left[i] - right[i]));
  }
  return result;
}

double RelativeCoefficientTwoNorm(const std::vector<double>& initial,
                                  const std::vector<double>& current)
{
  if (initial.size() != current.size() || !FiniteVector(initial) ||
      !FiniteVector(current))
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  long double difference_squared = 0.0L, initial_squared = 0.0L;
  for (std::size_t index = 0; index < initial.size(); ++index)
  {
    const long double difference = static_cast<long double>(current[index]) -
                                   initial[index];
    difference_squared += difference * difference;
    initial_squared += static_cast<long double>(initial[index]) * initial[index];
  }
  return std::sqrt(static_cast<double>(difference_squared)) /
         std::max(std::sqrt(static_cast<double>(initial_squared)), 1.0e-300);
}

double ConcatenatedCoefficientRelativeTwoNorm(
    const std::vector<double>& initial_interior,
    const std::vector<double>& final_interior,
    const std::vector<double>& initial_trace,
    const std::vector<double>& final_trace)
{
  if (initial_interior.size() != final_interior.size() ||
      initial_trace.size() != final_trace.size() ||
      !FiniteVector(initial_interior) || !FiniteVector(final_interior) ||
      !FiniteVector(initial_trace) || !FiniteVector(final_trace))
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  long double difference_squared = 0.0L, initial_squared = 0.0L;
  const auto accumulate = [&](const std::vector<double>& initial,
                              const std::vector<double>& current)
  {
    for (std::size_t index = 0; index < initial.size(); ++index)
    {
      const long double difference = static_cast<long double>(current[index]) -
                                     initial[index];
      difference_squared += difference * difference;
      initial_squared += static_cast<long double>(initial[index]) * initial[index];
    }
  };
  accumulate(initial_interior, final_interior);
  accumulate(initial_trace, final_trace);
  return std::sqrt(static_cast<double>(difference_squared)) /
         std::max(std::sqrt(static_cast<double>(initial_squared)), 1.0e-300);
}

void NormalizeInfinity(std::vector<double>& values)
{
  const double norm = InfinityNorm(values);
  if (norm > 0.0)
    for (double& value : values) { value /= norm; }
}

int OracleDecodeSignedVDof(const int signed_vdof, double& sign)
{
  sign = signed_vdof >= 0 ? 1.0 : -1.0;
  return signed_vdof >= 0 ? signed_vdof : -1 - signed_vdof;
}

bool OracleFlux(const ConservedState& state, const std::array<double, 2>& normal,
                ConservedState& flux)
{
  // Deliberately independent of production NormalFlux/Jacobian helpers.
  if (!(state[0] > 0.0)) return false;
  const double kinetic = 0.5 * (state[1] * state[1] + state[2] * state[2]) /
                         state[0];
  const double pressure = (gamma - 1.0) * (state[3] - kinetic);
  if (!(pressure > 0.0) || !std::isfinite(pressure)) return false;
  const double normal_velocity =
      (state[1] * normal[0] + state[2] * normal[1]) / state[0];
  flux = {{state[0] * normal_velocity,
           state[1] * normal_velocity + pressure * normal[0],
           state[2] * normal_velocity + pressure * normal[1],
           (state[3] + pressure) * normal_velocity}};
  return std::all_of(flux.begin(), flux.end(),
                     [](const double value) { return std::isfinite(value); });
}

ConservedState OracleManufacturedSource(const double x, const double y)
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double sx = std::sin(2.0 * pi * x), sy = std::sin(2.0 * pi * y);
  const double cx = std::cos(2.0 * pi * x), cy = std::cos(2.0 * pi * y);
  const double rho = 1.0 + 0.1 * sx * sy;
  const double u = 0.35 + 0.05 * cx * sy;
  const double v = -0.20 + 0.05 * sx * cy;
  const double pressure = 1.0 + 0.1 * cx * cy;
  const double rho_x = 0.2 * pi * cx * sy;
  const double rho_y = 0.2 * pi * sx * cy;
  const double u_x = -0.1 * pi * sx * sy;
  const double u_y = 0.1 * pi * cx * cy;
  const double v_x = 0.1 * pi * cx * cy;
  const double v_y = -0.1 * pi * sx * sy;
  const double pressure_x = -0.2 * pi * sx * cy;
  const double pressure_y = -0.2 * pi * cx * sy;
  const double energy = pressure / (gamma - 1.0) +
      0.5 * rho * (u * u + v * v);
  const double energy_x = pressure_x / (gamma - 1.0) +
      0.5 * rho_x * (u * u + v * v) + rho * (u * u_x + v * v_x);
  const double energy_y = pressure_y / (gamma - 1.0) +
      0.5 * rho_y * (u * u + v * v) + rho * (u * u_y + v * v_y);
  const double rhou_x = rho_x * u + rho * u_x;
  const double rhov_y = rho_y * v + rho * v_y;
  return {{rhou_x + rhov_y,
           rho_x * u * u + 2.0 * rho * u * u_x + pressure_x +
               rho_y * u * v + rho * u_y * v + rho * u * v_y,
           rho_x * u * v + rho * u_x * v + rho * u * v_x +
               rho_y * v * v + 2.0 * rho * v * v_y + pressure_y,
           (energy_x + pressure_x) * u + (energy + pressure) * u_x +
               (energy_y + pressure_y) * v + (energy + pressure) * v_y}};
}

bool InterpolateOracleState(const std::vector<double>& coefficients,
                            const int scalar_dofs, const int offset,
                            const mfem::Vector& shape, ConservedState& state)
{
  if (shape.Size() != scalar_dofs || offset < 0 ||
      offset + conserved_components * scalar_dofs >
          static_cast<int>(coefficients.size())) return false;
  state = {};
  for (int component = 0; component < conserved_components; ++component)
    for (int node = 0; node < scalar_dofs; ++node)
    {
      state[component] += shape[node] *
          coefficients[offset + node + scalar_dofs * component];
    }
  return std::all_of(state.begin(), state.end(),
                     [](const double value) { return std::isfinite(value); });
}

struct OrientationCase
{
  int face = -1;
  int element_one = -1;
  int element_two = -1;
  int element_one_orientation = -1;
  int element_two_orientation = -1;
};

enum class OrientationMutation
{
  none,
  wrong_elem2_normal,
  wrong_elem2_permutation
};

int FindElementFaceOffset(mfem::Mesh& mesh,
                          const mfem::FiniteElementSpace& trace_space,
                          const int element, const int selected_face)
{
  mfem::Array<int> faces, orientations;
  mesh.GetElementEdges(element, faces, orientations);
  int offset = 0;
  for (int local_face = 0; local_face < faces.Size(); ++local_face)
  {
    mfem::Array<int> face_vdofs;
    trace_space.GetFaceVDofs(faces[local_face], face_vdofs);
    if (faces[local_face] == selected_face) return offset;
    offset += face_vdofs.Size();
  }
  return -1;
}

bool ExtractElementCoefficients(const SmoothEulerExperiment& experiment,
                                const int element, std::vector<double>& interior,
                                std::vector<double>& trace)
{
  const mfem::GridFunction* interior_field = experiment.old_interior();
  const mfem::GridFunction* trace_field = experiment.old_trace();
  const mfem::FiniteElementSpace* trace_space = experiment.trace_space();
  if (interior_field == nullptr || trace_field == nullptr || trace_space == nullptr ||
      interior_field->FESpace() == nullptr || trace_space->GetMesh() == nullptr)
  {
    return false;
  }
  mfem::Array<int> interior_vdofs;
  interior_field->FESpace()->GetElementVDofs(element, interior_vdofs);
  interior.resize(interior_vdofs.Size());
  for (int local = 0; local < interior_vdofs.Size(); ++local)
  {
    double sign = 1.0;
    const int vdof = OracleDecodeSignedVDof(interior_vdofs[local], sign);
    if (vdof < 0 || vdof >= interior_field->Size()) return false;
    interior[local] = sign * (*interior_field)[vdof];
  }
  mfem::Array<int> faces, orientations;
  trace_space->GetMesh()->GetElementEdges(element, faces, orientations);
  trace.clear();
  for (int local_face = 0; local_face < faces.Size(); ++local_face)
  {
    mfem::Array<int> face_vdofs;
    trace_space->GetFaceVDofs(faces[local_face], face_vdofs);
    for (int local = 0; local < face_vdofs.Size(); ++local)
    {
      double sign = 1.0;
      const int vdof = OracleDecodeSignedVDof(face_vdofs[local], sign);
      if (vdof < 0 || vdof >= trace_field->Size()) return false;
      trace.push_back(sign * (*trace_field)[vdof]);
    }
  }
  return true;
}

bool AssembleIndependentQOnSpaces(
    const mfem::FiniteElementSpace& interior_space,
    const mfem::FiniteElementSpace& trace_space, const int quadrature_order,
    const int element, const std::vector<double>& interior,
    const std::vector<double>& base_interior, const std::vector<double>& trace,
    std::vector<double>& interior_residual,
    std::vector<double>& trace_residual,
    const OrientationCase* orientation = nullptr,
    const OrientationMutation mutation = OrientationMutation::none)
{
  // Reintegrate nonlinear Q directly from coefficients, FE shapes, geometry,
  // and explicit quadrature.  No production residual/Jacobian/condensation,
  // snapshot-population, or orientation helper is called here.
  if (interior_space.GetMesh() == nullptr || trace_space.GetMesh() == nullptr ||
      interior_space.GetMesh() != trace_space.GetMesh()) return false;
  mfem::Mesh& mesh = *trace_space.GetMesh();
  const mfem::FiniteElement* fe = interior_space.GetFE(element);
  mfem::ElementTransformation* transformation = mesh.GetElementTransformation(element);
  if (fe == nullptr || transformation == nullptr) return false;
  const int scalar_dofs = fe->GetDof();
  const int ni = conserved_components * scalar_dofs;
  if (static_cast<int>(interior.size()) != ni ||
      static_cast<int>(base_interior.size()) != ni) return false;
  interior_residual.assign(ni, 0.0);
  trace_residual.assign(trace.size(), 0.0);

  const mfem::IntegrationRule& volume_rule =
      mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
  mfem::Vector shape(scalar_dofs);
  mfem::DenseMatrix derivative(scalar_dofs, dimension);
  for (int point_index = 0; point_index < volume_rule.GetNPoints(); ++point_index)
  {
    const mfem::IntegrationPoint& point = volume_rule.IntPoint(point_index);
    transformation->SetIntPoint(&point);
    fe->CalcShape(point, shape);
    fe->CalcPhysDShape(*transformation, derivative);
    ConservedState state{}, old_state{}, flux_x{}, flux_y{};
    if (!InterpolateOracleState(interior, scalar_dofs, 0, shape, state) ||
        !InterpolateOracleState(base_interior, scalar_dofs, 0, shape, old_state) ||
        !OracleFlux(state, {{1.0, 0.0}}, flux_x) ||
        !OracleFlux(state, {{0.0, 1.0}}, flux_y)) return false;
    mfem::Vector physical(dimension);
    transformation->Transform(point, physical);
    const ConservedState source = OracleManufacturedSource(physical[0], physical[1]);
    const double weight = point.weight * transformation->Weight();
    for (int row = 0; row < scalar_dofs; ++row)
      for (int component = 0; component < conserved_components; ++component)
      {
        const int local_row = row + scalar_dofs * component;
        interior_residual[local_row] += weight *
            (shape[row] * (state[component] - old_state[component]) /
                 backward_euler_dt -
             derivative(row, 0) * flux_x[component] -
             derivative(row, 1) * flux_y[component] -
             shape[row] * source[component]);
      }
  }

  mfem::Array<int> faces, orientations;
  mesh.GetElementEdges(element, faces, orientations);
  int trace_offset = 0;
  for (int local_face = 0; local_face < faces.Size(); ++local_face)
  {
    const int face = faces[local_face];
    mfem::FaceElementTransformations* face_transform =
        mesh.GetFaceElementTransformations(face);
    const mfem::FiniteElement* trace_fe = trace_space.GetFaceElement(face);
    mfem::Array<int> face_vdofs;
    trace_space.GetFaceVDofs(face, face_vdofs);
    if (face_transform == nullptr || face_transform->Face == nullptr ||
        trace_fe == nullptr ||
        face_vdofs.Size() != conserved_components * trace_fe->GetDof() ||
        trace_offset + face_vdofs.Size() > static_cast<int>(trace.size())) return false;
    const bool is_element_one = face_transform->Elem1No == element;
    if (!is_element_one && face_transform->Elem2No != element) return false;
    const int trace_scalar_dofs = trace_fe->GetDof();
    const mfem::IntegrationRule& face_rule =
        mfem::IntRules.Get(trace_fe->GetGeomType(), quadrature_order);
    mfem::Vector interior_shape(scalar_dofs), trace_shape(trace_scalar_dofs);
    for (int point_index = 0; point_index < face_rule.GetNPoints(); ++point_index)
    {
      const mfem::IntegrationPoint& face_point = face_rule.IntPoint(point_index);
      face_transform->SetAllIntPoints(&face_point);
      const mfem::IntegrationPoint& element_point = is_element_one ?
          face_transform->GetElement1IntPoint() :
          face_transform->GetElement2IntPoint();
      fe->CalcShape(element_point, interior_shape);
      bool explicit_orientation = orientation != nullptr && face == orientation->face;
      bool reverse = false;
      if (explicit_orientation)
      {
        const int side_orientation = is_element_one ?
            orientation->element_one_orientation :
            orientation->element_two_orientation;
        if (side_orientation != 0 && side_orientation != 1) return false;
        mfem::IntegrationPoint side_point = face_point;
        side_point.x = side_orientation == 0 ? face_point.x : 1.0 - face_point.x;
        trace_fe->CalcShape(side_point, trace_shape);
        // The wrong-permutation mutation retains the oriented side point and
        // suppresses only the p-i coefficient/residual-row map.
        const bool use_wrong_elem2_permutation = !is_element_one &&
            mutation == OrientationMutation::wrong_elem2_permutation;
        reverse = side_orientation == 1 && !use_wrong_elem2_permutation;
      }
      else
      {
        trace_fe->CalcShape(face_point, trace_shape);
      }
      ConservedState interior_state{}, trace_state{};
      if (!InterpolateOracleState(interior, scalar_dofs, 0, interior_shape,
                                  interior_state)) return false;
      trace_state = {};
      for (int component = 0; component < conserved_components; ++component)
        for (int node = 0; node < trace_scalar_dofs; ++node)
        {
          const int canonical_node = reverse ? trace_scalar_dofs - 1 - node : node;
          trace_state[component] += trace_shape[node] * trace[
              trace_offset + canonical_node + trace_scalar_dofs * component];
        }
      mfem::Vector scaled_normal(dimension);
      mfem::CalcOrtho(face_transform->Face->Jacobian(), scaled_normal);
      const double face_measure = scaled_normal.Norml2();
      if (!(face_measure > 0.0)) return false;
      double side = is_element_one ? 1.0 : -1.0;
      if (explicit_orientation && !is_element_one &&
          mutation == OrientationMutation::wrong_elem2_normal) side = 1.0;
      const std::array<double, 2> normal{{side * scaled_normal[0] / face_measure,
                                          side * scaled_normal[1] / face_measure}};
      ConservedState flux{};
      if (!OracleFlux(trace_state, normal, flux)) return false;
      const double weight = face_point.weight * face_measure;
      for (int component = 0; component < conserved_components; ++component)
      {
        const double numerical_flux = flux[component] + fixed_scalar_tau *
            (interior_state[component] - trace_state[component]);
        for (int row = 0; row < scalar_dofs; ++row)
          interior_residual[row + scalar_dofs * component] +=
              weight * interior_shape[row] * numerical_flux;
        for (int row = 0; row < trace_scalar_dofs; ++row)
        {
          const int canonical_row = reverse ? trace_scalar_dofs - 1 - row : row;
          trace_residual[trace_offset + canonical_row +
                         trace_scalar_dofs * component] +=
              weight * trace_shape[row] * numerical_flux;
        }
      }
    }
    trace_offset += face_vdofs.Size();
  }
  return trace_offset == static_cast<int>(trace.size());
}

using OracleFluxMatrix =
    std::array<std::array<double, conserved_components>, conserved_components>;

bool OracleFluxJacobian(const ConservedState& state,
                        const std::array<double, dimension>& normal,
                        OracleFluxMatrix& jacobian)
{
  if (!std::all_of(state.begin(), state.end(),
                   [](const double value) { return std::isfinite(value); }) ||
      !std::isfinite(normal[0]) || !std::isfinite(normal[1]) ||
      !(state[0] > 0.0))
  {
    return false;
  }
  const double rho = state[0], mx = state[1], my = state[2], energy = state[3];
  const double momentum_normal = normal[0] * mx + normal[1] * my;
  const double normal_velocity = momentum_normal / rho;
  const double pressure = (gamma - 1.0) *
      (energy - 0.5 * (mx * mx + my * my) / rho);
  if (!(pressure > 0.0) || !std::isfinite(pressure)) return false;
  const std::array<double, conserved_components> dp{{
      0.5 * (gamma - 1.0) * (mx * mx + my * my) / (rho * rho),
      -(gamma - 1.0) * mx / rho,
      -(gamma - 1.0) * my / rho,
      gamma - 1.0}};
  const std::array<double, conserved_components> dvn{{
      -momentum_normal / (rho * rho), normal[0] / rho,
      normal[1] / rho, 0.0}};
  jacobian = {};
  jacobian[0][1] = normal[0];
  jacobian[0][2] = normal[1];
  for (int column = 0; column < conserved_components; ++column)
  {
    jacobian[1][column] = (column == 1 ? normal_velocity : 0.0) +
        mx * dvn[column] + normal[0] * dp[column];
    jacobian[2][column] = (column == 2 ? normal_velocity : 0.0) +
        my * dvn[column] + normal[1] * dp[column];
    jacobian[3][column] =
        ((column == 3 ? 1.0 : 0.0) + dp[column]) * normal_velocity +
        (energy + pressure) * dvn[column];
  }
  for (const auto& row : jacobian)
    for (const double value : row)
      if (!std::isfinite(value)) return false;
  return true;
}

bool AssembleIndependentAnalyticBlocks(
    const mfem::FiniteElementSpace& interior_space,
    const mfem::FiniteElementSpace& trace_space, const int quadrature_order,
    const int element, const std::vector<double>& interior,
    const std::vector<double>& trace, std::vector<double>& a,
    std::vector<double>& b, std::vector<double>& c, std::vector<double>& d)
{
  if (interior_space.GetMesh() == nullptr ||
      interior_space.GetMesh() != trace_space.GetMesh()) return false;
  mfem::Mesh& mesh = *interior_space.GetMesh();
  const mfem::FiniteElement* fe = interior_space.GetFE(element);
  mfem::ElementTransformation* transformation =
      mesh.GetElementTransformation(element);
  if (fe == nullptr || transformation == nullptr) return false;
  const int scalar_dofs = fe->GetDof();
  const int ni = conserved_components * scalar_dofs;
  if (static_cast<int>(interior.size()) != ni) return false;

  mfem::Array<int> faces, orientations;
  mesh.GetElementEdges(element, faces, orientations);
  int nt = 0;
  for (int local_face = 0; local_face < faces.Size(); ++local_face)
  {
    mfem::Array<int> face_vdofs;
    trace_space.GetFaceVDofs(faces[local_face], face_vdofs);
    nt += face_vdofs.Size();
  }
  if (static_cast<int>(trace.size()) != nt) return false;
  a.assign(static_cast<std::size_t>(ni) * ni, 0.0);
  b.assign(static_cast<std::size_t>(ni) * nt, 0.0);
  c.assign(static_cast<std::size_t>(nt) * ni, 0.0);
  d.assign(static_cast<std::size_t>(nt) * nt, 0.0);

  const mfem::IntegrationRule& volume_rule =
      mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
  mfem::Vector shape(scalar_dofs);
  mfem::DenseMatrix derivative(scalar_dofs, dimension);
  for (int q = 0; q < volume_rule.GetNPoints(); ++q)
  {
    const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
    transformation->SetIntPoint(&point);
    fe->CalcShape(point, shape);
    fe->CalcPhysDShape(*transformation, derivative);
    ConservedState state{};
    OracleFluxMatrix x_jacobian{}, y_jacobian{};
    if (!InterpolateOracleState(interior, scalar_dofs, 0, shape, state) ||
        !OracleFluxJacobian(state, {{1.0, 0.0}}, x_jacobian) ||
        !OracleFluxJacobian(state, {{0.0, 1.0}}, y_jacobian)) return false;
    const double weight = point.weight * transformation->Weight();
    if (!(weight > 0.0) || !std::isfinite(weight)) return false;
    for (int row_node = 0; row_node < scalar_dofs; ++row_node)
      for (int column_node = 0; column_node < scalar_dofs; ++column_node)
        for (int row_component = 0; row_component < conserved_components;
             ++row_component)
          for (int column_component = 0;
               column_component < conserved_components; ++column_component)
          {
            const int row = row_node + scalar_dofs * row_component;
            const int column = column_node + scalar_dofs * column_component;
            const double mass = row_component == column_component ?
                shape[row_node] * shape[column_node] / backward_euler_dt : 0.0;
            a[static_cast<std::size_t>(row) * ni + column] += weight *
                (mass - derivative(row_node, 0) *
                            x_jacobian[row_component][column_component] -
                        derivative(row_node, 1) *
                            y_jacobian[row_component][column_component]);
          }
  }

  int trace_offset = 0;
  for (int local_face = 0; local_face < faces.Size(); ++local_face)
  {
    const int face = faces[local_face];
    mfem::FaceElementTransformations* face_transform =
        mesh.GetFaceElementTransformations(face);
    const mfem::FiniteElement* trace_fe = trace_space.GetFaceElement(face);
    mfem::Array<int> face_vdofs;
    trace_space.GetFaceVDofs(face, face_vdofs);
    if (face_transform == nullptr || face_transform->Face == nullptr ||
        trace_fe == nullptr) return false;
    const bool first = face_transform->Elem1No == element;
    if (!first && face_transform->Elem2No != element) return false;
    const int trace_scalar_dofs = trace_fe->GetDof();
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(trace_fe->GetGeomType(), quadrature_order);
    mfem::Vector interior_shape(scalar_dofs), trace_shape(trace_scalar_dofs);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& face_point = rule.IntPoint(q);
      face_transform->SetAllIntPoints(&face_point);
      const mfem::IntegrationPoint& element_point = first ?
          face_transform->GetElement1IntPoint() :
          face_transform->GetElement2IntPoint();
      fe->CalcShape(element_point, interior_shape);
      trace_fe->CalcShape(face_point, trace_shape);
      ConservedState trace_state{};
      if (!InterpolateOracleState(trace, trace_scalar_dofs, trace_offset,
                                  trace_shape, trace_state)) return false;
      mfem::Vector scaled_normal(dimension);
      mfem::CalcOrtho(face_transform->Face->Jacobian(), scaled_normal);
      const double face_measure = scaled_normal.Norml2();
      if (!(face_measure > 0.0) || !std::isfinite(face_measure)) return false;
      const double side = first ? 1.0 : -1.0;
      const std::array<double, dimension> normal{{
          side * scaled_normal[0] / face_measure,
          side * scaled_normal[1] / face_measure}};
      OracleFluxMatrix trace_jacobian{};
      if (!OracleFluxJacobian(trace_state, normal, trace_jacobian)) return false;
      const double weight = face_point.weight * face_measure;
      for (int row_component = 0; row_component < conserved_components;
           ++row_component)
        for (int column_component = 0;
             column_component < conserved_components; ++column_component)
        {
          const double interior_derivative =
              row_component == column_component ? fixed_scalar_tau : 0.0;
          const double trace_derivative =
              trace_jacobian[row_component][column_component] -
              (row_component == column_component ? fixed_scalar_tau : 0.0);
          for (int row_node = 0; row_node < scalar_dofs; ++row_node)
          {
            const int interior_row = row_node + scalar_dofs * row_component;
            for (int column_node = 0; column_node < scalar_dofs; ++column_node)
            {
              const int interior_column =
                  column_node + scalar_dofs * column_component;
              a[static_cast<std::size_t>(interior_row) * ni + interior_column] +=
                  weight * interior_shape[row_node] *
                  interior_shape[column_node] * interior_derivative;
            }
            for (int column_node = 0; column_node < trace_scalar_dofs;
                 ++column_node)
            {
              const int trace_column = trace_offset + column_node +
                  trace_scalar_dofs * column_component;
              b[static_cast<std::size_t>(interior_row) * nt + trace_column] +=
                  weight * interior_shape[row_node] * trace_shape[column_node] *
                  trace_derivative;
            }
          }
          for (int row_node = 0; row_node < trace_scalar_dofs; ++row_node)
          {
            const int trace_row = trace_offset + row_node +
                trace_scalar_dofs * row_component;
            for (int column_node = 0; column_node < scalar_dofs; ++column_node)
            {
              const int interior_column =
                  column_node + scalar_dofs * column_component;
              c[static_cast<std::size_t>(trace_row) * ni + interior_column] +=
                  weight * trace_shape[row_node] * interior_shape[column_node] *
                  interior_derivative;
            }
            for (int column_node = 0; column_node < trace_scalar_dofs;
                 ++column_node)
            {
              const int trace_column = trace_offset + column_node +
                  trace_scalar_dofs * column_component;
              d[static_cast<std::size_t>(trace_row) * nt + trace_column] +=
                  weight * trace_shape[row_node] * trace_shape[column_node] *
                  trace_derivative;
            }
          }
        }
    }
    trace_offset += face_vdofs.Size();
  }
  return trace_offset == nt && FiniteVector(a) && FiniteVector(b) &&
         FiniteVector(c) && FiniteVector(d);
}

bool AssembleIndependentQ(const SmoothEulerExperiment& experiment,
                          const int element, const std::vector<double>& interior,
                          const std::vector<double>& base_interior,
                          const std::vector<double>& trace,
                          std::vector<double>& interior_residual,
                          std::vector<double>& trace_residual,
                          const OrientationCase* orientation = nullptr,
                          const OrientationMutation mutation = OrientationMutation::none)
{
  const mfem::GridFunction* interior_field = experiment.old_interior();
  const mfem::FiniteElementSpace* trace_space = experiment.trace_space();
  if (interior_field == nullptr || interior_field->FESpace() == nullptr ||
      trace_space == nullptr) return false;
  return AssembleIndependentQOnSpaces(
      *interior_field->FESpace(), *trace_space, experiment.quadrature_order(),
      element, interior, base_interior, trace, interior_residual,
      trace_residual, orientation, mutation);
}

std::vector<double> ApplySnapshotBlock(const std::vector<double>& block,
                                       const int element, const int rows,
                                       const int columns,
                                       const std::vector<double>& direction)
{
  if (static_cast<int>(direction.size()) != columns ||
      block.size() < static_cast<std::size_t>(element + 1) * rows * columns)
  {
    return {};
  }
  std::vector<double> action(rows, 0.0);
  const std::size_t offset = static_cast<std::size_t>(element) * rows * columns;
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < columns; ++column)
    {
      action[row] += block[offset + static_cast<std::size_t>(row) * columns +
                           column] * direction[column];
    }
  return action;
}

bool SnapshotHasExactExtents(const LocalLinearizationSnapshot& snapshot)
{
  if (snapshot.elements <= 0 || snapshot.interior_dofs <= 0 ||
      snapshot.trace_dofs <= 0) return false;
  const std::size_t elements = static_cast<std::size_t>(snapshot.elements);
  const std::size_t ni = static_cast<std::size_t>(snapshot.interior_dofs);
  const std::size_t nt = static_cast<std::size_t>(snapshot.trace_dofs);
  return snapshot.a.size() == elements * ni * ni &&
         snapshot.b.size() == elements * ni * nt &&
         snapshot.c.size() == elements * nt * ni &&
         snapshot.d.size() == elements * nt * nt &&
         snapshot.f.size() == elements * ni &&
         snapshot.g.size() == elements * nt &&
         snapshot.signed_trace_dofs.size() == elements * nt;
}

bool FindTwoSidedOrientationCase(const SmoothEulerExperiment& experiment,
                                 OrientationCase& selected)
{
  selected = {};
  const mfem::FiniteElementSpace* trace_space = experiment.trace_space();
  if (trace_space == nullptr || trace_space->GetMesh() == nullptr) return false;
  mfem::Mesh& mesh = *trace_space->GetMesh();
  int faces = 0;
  int two_sided = 0;
  int p2 = 0;
  int orientation_0_1 = 0;
  int ordinary = 0;
  int coincident_faces = 0;
  int admitted = 0;
  int periodic_excluded = 0;
  int rejected_not_local_conforming = 0;
  int rejected_invalid_elements = 0;
  int rejected_not_p2_segment = 0;
  int rejected_orientation = 0;
  int rejected_invalid_transform = 0;
  int rejected_not_coincident = 0;
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    ++faces;
    const mfem::Mesh::FaceInformation face_information =
        mesh.GetFaceInformation(face);
    if (face_information.tag != mfem::Mesh::FaceInfoTag::LocalConforming ||
        !face_information.IsLocal() || !face_information.IsConforming())
    {
      ++rejected_not_local_conforming;
      continue;
    }
    const int element_one = face_information.element[0].index;
    const int element_two = face_information.element[1].index;
    if (element_one < 0 || element_one >= mesh.GetNE() || element_two < 0 ||
        element_two >= mesh.GetNE() || element_one == element_two)
    {
      ++rejected_invalid_elements;
      continue;
    }
    ++two_sided;
    const mfem::FiniteElement* face_fe = trace_space->GetFaceElement(face);
    if (face_fe == nullptr || face_fe->GetGeomType() != mfem::Geometry::SEGMENT ||
        face_fe->GetOrder() != 2 || face_fe->GetDof() != 3)
    {
      ++rejected_not_p2_segment;
      continue;
    }
    ++p2;
    const int element_one_orientation = face_information.element[0].orientation;
    const int element_two_orientation = face_information.element[1].orientation;
    if (element_one_orientation != 0 || element_two_orientation != 1)
    {
      ++rejected_orientation;
      continue;
    }
    ++orientation_0_1;
    mfem::FaceElementTransformations* transform =
        mesh.GetFaceElementTransformations(face);
    constexpr int required_face_configuration =
        mfem::FaceElementTransformations::HAVE_ELEM1 |
        mfem::FaceElementTransformations::HAVE_ELEM2 |
        mfem::FaceElementTransformations::HAVE_LOC1 |
        mfem::FaceElementTransformations::HAVE_LOC2 |
        mfem::FaceElementTransformations::HAVE_FACE;
    // MFEM pointer-returning mesh transformation access reuses scratch
    // Transformation storage, so it cannot occur while this face bundle is live.
    if (transform == nullptr ||
        (transform->GetConfigurationMask() & required_face_configuration) !=
            required_face_configuration ||
        transform->Face == nullptr || transform->Elem1 == nullptr ||
        transform->Elem2 == nullptr || transform->Elem1 == transform->Elem2 ||
        transform->Elem1No != element_one || transform->Elem2No != element_two ||
        transform->Elem1->ElementNo != element_one ||
        transform->Elem2->ElementNo != element_two)
    {
      ++rejected_invalid_transform;
      continue;
    }
    mfem::IntegrationPoint midpoint;
    midpoint.x = 0.5;
    transform->SetAllIntPoints(&midpoint);
    mfem::Vector face_physical(dimension), one_physical(dimension),
        two_physical(dimension);
    transform->Face->Transform(midpoint, face_physical);
    transform->Elem1->Transform(transform->GetElement1IntPoint(), one_physical);
    transform->Elem2->Transform(transform->GetElement2IntPoint(), two_physical);
    // Both midpoint coordinates strictly inside the unit square exclude the
    // periodic seams; the generated Cartesian mesh is ordinary conforming.
    if (!(face_physical[0] > 1.0e-12 && face_physical[0] < 1.0 - 1.0e-12 &&
          face_physical[1] > 1.0e-12 && face_physical[1] < 1.0 - 1.0e-12))
    {
      ++periodic_excluded;
      continue;
    }
    ++ordinary;
    bool coincident = true;
    const mfem::IntegrationRule& coincidence_rule = mfem::IntRules.Get(
        face_fe->GetGeomType(), experiment.quadrature_order());
    for (int node = 0; node < coincidence_rule.GetNPoints(); ++node)
    {
      const mfem::IntegrationPoint& point = coincidence_rule.IntPoint(node);
      transform->SetAllIntPoints(&point);
      transform->Face->Transform(point, face_physical);
      transform->Elem1->Transform(transform->GetElement1IntPoint(), one_physical);
      transform->Elem2->Transform(transform->GetElement2IntPoint(), two_physical);
      for (int coordinate = 0; coordinate < dimension; ++coordinate)
      {
        coincident = coincident &&
            std::abs(face_physical[coordinate] - one_physical[coordinate]) <= 2.0e-14 &&
            std::abs(face_physical[coordinate] - two_physical[coordinate]) <= 2.0e-14;
      }
    }
    if (!coincident)
    {
      ++rejected_not_coincident;
      continue;
    }
    ++coincident_faces;
    ++admitted;
    // The ascending scan fixes the first admitted face but continues so the
    // topology census remains deterministic and non-vacuous.
    if (selected.face < 0)
    {
      selected.face = face;
      selected.element_one = element_one;
      selected.element_two = element_two;
      selected.element_one_orientation = element_one_orientation;
      selected.element_two_orientation = element_two_orientation;
    }
  }
  const bool exact_census = faces == 18 && two_sided == 18 && p2 == 18 &&
      orientation_0_1 == 18 && ordinary == 12 && coincident_faces == 12 &&
      admitted == 12 && periodic_excluded == 6;
  if (!exact_census || admitted == 0 || selected.face < 0)
  {
    std::cerr << "DETAIL predicate=two_sided_p2_orientation_selector"
              << " faces=" << faces
              << " two_sided=" << two_sided
              << " p2=" << p2
              << " orientation_0_1=" << orientation_0_1
              << " ordinary=" << ordinary
              << " coincident=" << coincident_faces
              << " admitted=" << admitted
              << " periodic_excluded=" << periodic_excluded
              << " rejected_not_local_conforming="
              << rejected_not_local_conforming
              << " rejected_invalid_elements=" << rejected_invalid_elements
              << " rejected_not_p2_segment=" << rejected_not_p2_segment
              << " rejected_orientation=" << rejected_orientation
              << " rejected_invalid_transform=" << rejected_invalid_transform
              << " rejected_not_coincident=" << rejected_not_coincident
              << " selected=(face=" << selected.face
              << ",elements=(" << selected.element_one << ','
              << selected.element_two << "),orientations=("
              << selected.element_one_orientation << ','
              << selected.element_two_orientation << "))\n";
    return false;
  }
  return true;
}

double NormalizedTraceActionError(const std::vector<double>& interior_action,
                                  const std::vector<double>& trace_action,
                                  const std::vector<double>& expected_interior,
                                  const std::vector<double>& expected_trace)
{
  const double scale = std::max({1.0, InfinityNorm(interior_action),
      InfinityNorm(trace_action), InfinityNorm(expected_interior),
      InfinityNorm(expected_trace)});
  return std::max(DifferenceNorm(interior_action, expected_interior),
                  DifferenceNorm(trace_action, expected_trace)) / scale;
}

int TestHandFluxAndFailureAtomicity()
{
  const ConservedState state{{1.0, 2.0, 3.0, 10.0}};
  FluxVector flux{{-7.0, -7.0, -7.0, -7.0}};
  if (!NormalFlux(state, {{1.0, 0.0}}, flux).ok()) { return 1; }
  const FluxVector expected{{2.0, 5.4, 6.0, 22.8}};
  for (int component = 0; component < conserved_components; ++component)
  {
    if (!Close(flux[component], expected[component], 2.0e-14)) { return 2; }
  }
  const FluxVector saved = flux;
  ConservedState bad_density = state;
  bad_density[0] = 0.0;
  if (NormalFlux(bad_density, {{1.0, 0.0}}, flux).status != PhysicsStatus::invalid_density ||
      flux != saved) { return 3; }
  ConservedState bad_pressure = state;
  bad_pressure[3] = 6.5;
  if (NormalFlux(bad_pressure, {{1.0, 0.0}}, flux).status != PhysicsStatus::invalid_pressure ||
      flux != saved) { return 4; }
  ConservedState non_finite = state;
  non_finite[2] = std::numeric_limits<double>::infinity();
  if (NormalFlux(non_finite, {{1.0, 0.0}}, flux).status != PhysicsStatus::non_finite ||
      flux != saved) { return 5; }
  return 0;
}

int TestAnalyticFluxJacobian()
{
  const ConservedState state{{1.2, 0.35, -0.22, 2.8}};
  const std::array<double, 2> normal{{0.6, -0.8}};
  FluxJacobian analytic{};
  if (!NormalFluxJacobian(state, normal, analytic).ok()) { return 1; }
  if (analytic[0][0] != 0.0) { return 2; }
  for (int column = 0; column < conserved_components; ++column)
  {
    ConservedState plus = state;
    ConservedState minus = state;
    const double step = 1.0e-7 * std::max(1.0, std::abs(state[column]));
    plus[column] += step;
    minus[column] -= step;
    FluxVector positive{};
    FluxVector negative{};
    if (!NormalFlux(plus, normal, positive).ok() || !NormalFlux(minus, normal, negative).ok())
    {
      return 3;
    }
    for (int row = 0; row < conserved_components; ++row)
    {
      const double difference = (positive[row] - negative[row]) / (2.0 * step);
      if (!Close(analytic[row][column], difference, 2.0e-6)) { return 4; }
    }
  }
  return 0;
}

int TestManufacturedSource()
{
  constexpr double x = 0.173, y = 0.619, h = 1.0e-5;
  ConservedState source{};
  if (!ManufacturedSource(x, y, source).ok()) { return 1; }
  FluxVector xp{}, xm{}, yp{}, ym{};
  ConservedState state{};
  if (!ManufacturedConservative(x + h, y, state).ok() || !NormalFlux(state, {{1.0, 0.0}}, xp).ok() ||
      !ManufacturedConservative(x - h, y, state).ok() || !NormalFlux(state, {{1.0, 0.0}}, xm).ok() ||
      !ManufacturedConservative(x, y + h, state).ok() || !NormalFlux(state, {{0.0, 1.0}}, yp).ok() ||
      !ManufacturedConservative(x, y - h, state).ok() || !NormalFlux(state, {{0.0, 1.0}}, ym).ok())
  {
    return 2;
  }
  for (int component = 0; component < conserved_components; ++component)
  {
    const double finite_difference = (xp[component] - xm[component] + yp[component] - ym[component]) /
                                     (2.0 * h);
    if (!Close(source[component], finite_difference, 2.0e-6)) { return 3; }
  }
  const double maximum_wave_speed = ManufacturedMaximumWaveSpeed();
  return maximum_wave_speed < 1.78 && fixed_scalar_tau - maximum_wave_speed > 0.72 ? 0 : 4;
}

int TestMfemByNodesAndSignedVDofs()
{
  constexpr int scalar_count = 9;
  for (int component = 0; component < conserved_components; ++component)
  {
    for (int scalar = 0; scalar < scalar_count; ++scalar)
    {
      const int vector_dof = ByNodesIndex(scalar, scalar_count, component);
      const int mfem_vector_dof = mfem::Ordering::Map<mfem::Ordering::byNODES>(
          scalar_count, conserved_components, scalar, component);
      if (vector_dof != mfem_vector_dof ||
          ComponentFromByNodesIndex(vector_dof, scalar_count) != component ||
          ScalarFromByNodesIndex(vector_dof, scalar_count) != scalar)
      {
        return 1;
      }
    }
  }
  double positive_sign = 0.0;
  double production_positive_sign = 0.0;
  if (DecodeSignedVDof(23, production_positive_sign) !=
          mfem::FiniteElementSpace::DecodeDof(23, positive_sign) ||
      production_positive_sign != positive_sign)
  {
    return 2;
  }
  double negative_sign = 0.0;
  double production_negative_sign = 0.0;
  if (DecodeSignedVDof(-24, production_negative_sign) !=
          mfem::FiniteElementSpace::DecodeDof(-24, negative_sign) ||
      production_negative_sign != negative_sign)
  {
    return 3;
  }
  return 0;
}

int TestProjectedTraceComponentOrdering()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage())
  {
    return 1;
  }
  const mfem::FiniteElementSpace* trace_space = experiment.trace_space();
  const mfem::GridFunction* trace = experiment.old_trace();
  if (trace_space == nullptr || trace == nullptr || trace_space->GetMesh() == nullptr)
  {
    return 2;
  }
  mfem::Mesh* mesh = trace_space->GetMesh();
  for (int face = 0; face < mesh->GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* face_fe = trace_space->GetFaceElement(face);
    mfem::ElementTransformation* transformation = mesh->GetFaceTransformation(face);
    if (face_fe == nullptr || transformation == nullptr) { return 3; }
    const int face_scalar_dofs = face_fe->GetDof();
    mfem::Array<int> face_vdofs;
    trace_space->GetFaceVDofs(face, face_vdofs);
    if (face_vdofs.Size() != conserved_components * face_scalar_dofs) { return 4; }
    for (int scalar = 0; scalar < face_scalar_dofs; ++scalar)
    {
      mfem::Vector point(dg::hdg::euler::dimension);
      transformation->Transform(face_fe->GetNodes().IntPoint(scalar), point);
      ConservedState state{};
      if (!ManufacturedConservative(point[0], point[1], state).ok()) { return 5; }
      for (int component = 0; component < conserved_components; ++component)
      {
        const int local = ByNodesIndex(scalar, face_scalar_dofs, component);
        double production_sign = 0.0;
        const int vdof = DecodeSignedVDof(face_vdofs[local], production_sign);
        if (vdof < 0 || vdof >= trace->Size() ||
            !Close((*trace)[vdof], production_sign * state[component], 2.0e-14))
        {
          return 6;
        }
      }
    }
  }
  return 0;
}

int TestElementFaceOrientationAndPermutation()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage() ||
      !experiment.AssembleLinearization()) return 1;
  LocalLinearizationSnapshot snapshot;
  const mfem::FiniteElementSpace* trace_space = experiment.trace_space();
  if (trace_space == nullptr || trace_space->GetMesh() == nullptr ||
      !experiment.CopyLocalLinearization(snapshot)) return 2;
  mfem::Mesh* mesh = trace_space->GetMesh();
  int ordinary_zero_shift_samples = 0;
  int periodic_x_shift_samples = 0;
  int periodic_y_shift_samples = 0;
  double maximum_wrapped_error = 0.0;
  double maximum_unwrapped_seam_discrepancy = 0.0;
  for (int element = 0; element < snapshot.elements; ++element)
  {
    mfem::Array<int> faces, orientations;
    mesh->GetElementEdges(element, faces, orientations);
    int offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      const mfem::FiniteElement* face_fe = trace_space->GetFaceElement(face);
      mfem::FaceElementTransformations* transform = mesh->GetFaceElementTransformations(face);
      mfem::Array<int> face_vdofs;
      trace_space->GetFaceVDofs(face, face_vdofs);
      if (face_fe == nullptr || transform == nullptr ||
          (transform->Elem1No != element && transform->Elem2No != element)) return 3;
      for (int local = 0; local < face_vdofs.Size(); ++local)
      {
        if (snapshot.signed_trace_dofs[static_cast<std::size_t>(element) *
                                       snapshot.trace_dofs + offset + local] != face_vdofs[local])
        {
          return 4;
        }
      }
      for (int node = 0; node < face_fe->GetNodes().GetNPoints(); ++node)
      {
        const mfem::IntegrationPoint& face_point = face_fe->GetNodes().IntPoint(node);
        transform->SetAllIntPoints(&face_point);
        const bool element_is_elem1 = transform->Elem1No == element;
        const mfem::IntegrationPoint& element_point = element_is_elem1 ?
            transform->GetElement1IntPoint() : transform->GetElement2IntPoint();
        mfem::Vector face_physical(dg::hdg::euler::dimension);
        mfem::Vector element_physical(dg::hdg::euler::dimension);
        transform->Face->Transform(face_point, face_physical);
        mesh->GetElementTransformation(element)->Transform(element_point, element_physical);
        std::array<double, dimension> raw_delta{}, lattice_shift{}, wrapped_delta{};
        bool lattice_shift_valid = true;
        bool wrapped_close = true;
        for (int coordinate = 0; coordinate < dimension; ++coordinate)
        {
          raw_delta[coordinate] = element_physical[coordinate] -
                                  face_physical[coordinate];
          lattice_shift[coordinate] = std::round(raw_delta[coordinate]);
          wrapped_delta[coordinate] = raw_delta[coordinate] -
                                      lattice_shift[coordinate];
          lattice_shift_valid = lattice_shift_valid &&
              (lattice_shift[coordinate] == -1.0 ||
               lattice_shift[coordinate] == 0.0 ||
               lattice_shift[coordinate] == 1.0);
          wrapped_close = wrapped_close &&
              Close(wrapped_delta[coordinate], 0.0, 2.0e-14);
          maximum_wrapped_error = std::max(
              maximum_wrapped_error, std::abs(wrapped_delta[coordinate]));
        }
        const bool zero_shift = lattice_shift[0] == 0.0 &&
                                lattice_shift[1] == 0.0;
        const bool ordinary_raw_coordinates_close = !zero_shift ||
            (Close(face_physical[0], element_physical[0], 2.0e-14) &&
             Close(face_physical[1], element_physical[1], 2.0e-14));
        if (!lattice_shift_valid || !wrapped_close ||
            (element_is_elem1 && !zero_shift) ||
            !ordinary_raw_coordinates_close)
        {
          std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10)
                    << "DETAIL predicate=periodic_face_equivalence"
                    << " face=" << face
                    << " element=" << element
                    << " side=" << (element_is_elem1 ? "Elem1" : "Elem2")
                    << " node=" << node
                    << " face=(" << face_physical[0] << ',' << face_physical[1] << ')'
                    << " element=(" << element_physical[0] << ','
                    << element_physical[1] << ')'
                    << " raw_delta=(" << raw_delta[0] << ',' << raw_delta[1] << ')'
                    << " lattice_shift=(" << lattice_shift[0] << ','
                    << lattice_shift[1] << ')'
                    << " wrapped_delta=(" << wrapped_delta[0] << ','
                    << wrapped_delta[1] << ')'
                    << " tolerance=2e-14\n";
          return 5;
        }
        if (!element_is_elem1)
        {
          if (zero_shift) ++ordinary_zero_shift_samples;
          if (lattice_shift[0] != 0.0) ++periodic_x_shift_samples;
          if (lattice_shift[1] != 0.0) ++periodic_y_shift_samples;
          if (!zero_shift)
          {
            maximum_unwrapped_seam_discrepancy = std::max(
                maximum_unwrapped_seam_discrepancy,
                std::max(std::abs(raw_delta[0]), std::abs(raw_delta[1])));
          }
        }
      }
      offset += face_vdofs.Size();
    }
    if (offset != snapshot.trace_dofs || orientations.Size() != faces.Size()) return 6;
  }
  if (ordinary_zero_shift_samples == 0 || periodic_x_shift_samples == 0 ||
      periodic_y_shift_samples == 0 ||
      !(maximum_unwrapped_seam_discrepancy > 0.5))
  {
    std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10)
              << "DETAIL predicate=periodic_face_nonvacuity"
              << " ordinary_count=" << ordinary_zero_shift_samples
              << " x_shift_count=" << periodic_x_shift_samples
              << " y_shift_count=" << periodic_y_shift_samples
              << " maximum_wrapped_error=" << maximum_wrapped_error
              << " maximum_unwrapped_error="
              << maximum_unwrapped_seam_discrepancy
              << " unwrapped_threshold=0.5\n";
    return 5;
  }
  return 0;
}

int TestFailurePreservingSetup()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage())
  {
    return 1;
  }
  if (experiment.BuildPeriodicMesh(2)) { return 2; }
  // A failed fresh build cannot displace the previously complete mesh/space
  // graph, so the already configured projection remains reusable.
  if (!experiment.ProjectOldStage()) { return 3; }
  return experiment.BuildPeriodicMesh(3) && experiment.ProjectOldStage() ? 0 : 4;
}

int TestSnapshotGenerationAndExtentValidation()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage() ||
      !experiment.AssembleLinearization()) return 1;
  LocalLinearizationSnapshot first;
  if (!experiment.CopyLocalLinearization(first) || !SnapshotHasExactExtents(first))
  {
    return 2;
  }
  LocalLinearizationSnapshot stale = first;
  if (!experiment.ProjectOldStage()) return 3;
  if (experiment.CopyLocalLinearization(stale) || stale.elements != 0 ||
      !stale.a.empty() || !stale.b.empty() || !stale.c.empty() ||
      !stale.d.empty() || !stale.f.empty() || !stale.g.empty() ||
      !stale.signed_trace_dofs.empty()) return 4;
  LocalLinearizationSnapshot second;
  if (!experiment.AssembleLinearization() ||
      !experiment.CopyLocalLinearization(second) ||
      !SnapshotHasExactExtents(second)) return 5;

  // A successful graph rebuild destroys the prior generation.  Assembly
  // without a new projection fails, and no stale snapshot is exposed.
  if (!experiment.BuildPeriodicMesh(3)) return 6;
  stale = second;
  if (experiment.CopyLocalLinearization(stale) || stale.elements != 0) return 7;
  if (experiment.AssembleLinearization()) return 8;
  if (experiment.CopyLocalLinearization(stale) || stale.elements != 0) return 9;
  LocalLinearizationSnapshot third;
  if (!experiment.ProjectOldStage() || !experiment.AssembleLinearization() ||
      !experiment.CopyLocalLinearization(third) ||
      !SnapshotHasExactExtents(third)) return 10;
  return 0;
}

int TestIndependentAssembledLinearizationOracle()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage() ||
      !experiment.AssembleLinearization()) return 1;
  LocalLinearizationSnapshot snapshot;
  if (!experiment.CopyLocalLinearization(snapshot) ||
      !SnapshotHasExactExtents(snapshot)) return 2;
  constexpr double epsilon = experiment_epsilon;
  for (int element = 0; element < snapshot.elements; ++element)
  {
    std::vector<double> base_interior, base_trace;
    if (!ExtractElementCoefficients(experiment, element, base_interior, base_trace) ||
        static_cast<int>(base_interior.size()) != snapshot.interior_dofs ||
        static_cast<int>(base_trace.size()) != snapshot.trace_dofs) return 3;
    std::vector<double> residual_r, residual_g;
    if (!AssembleIndependentQ(experiment, element, base_interior, base_interior,
                              base_trace, residual_r, residual_g)) return 4;
    std::vector<double> expected_r(snapshot.interior_dofs),
                        expected_g(snapshot.trace_dofs);
    for (int row = 0; row < snapshot.interior_dofs; ++row)
      expected_r[row] = -snapshot.f[static_cast<std::size_t>(element) *
                                     snapshot.interior_dofs + row];
    for (int row = 0; row < snapshot.trace_dofs; ++row)
      expected_g[row] = -snapshot.g[static_cast<std::size_t>(element) *
                                     snapshot.trace_dofs + row];
    const double q_scale = std::max({1.0, InfinityNorm(residual_r),
        InfinityNorm(residual_g)});
    const double vector_tolerance = 2048.0 * epsilon * q_scale;
    if (DifferenceNorm(residual_r, expected_r) > vector_tolerance ||
        DifferenceNorm(residual_g, expected_g) > vector_tolerance) return 5;

    std::vector<double> interior_direction(snapshot.interior_dofs),
                        trace_direction(snapshot.trace_dofs);
    for (int i = 0; i < snapshot.interior_dofs; ++i)
      interior_direction[i] = std::sin(0.37 * (i + 1)) +
                              0.23 * std::cos(0.19 * (i + 2));
    for (int i = 0; i < snapshot.trace_dofs; ++i)
      trace_direction[i] = std::cos(0.29 * (i + 1)) -
                           0.31 * std::sin(0.17 * (i + 3));
    NormalizeInfinity(interior_direction);
    NormalizeInfinity(trace_direction);
    if (InfinityNorm(interior_direction) != 1.0 ||
        InfinityNorm(trace_direction) != 1.0) return 6;
    const double step = std::cbrt(epsilon) *
        std::max({1.0, InfinityNorm(base_interior), InfinityNorm(base_trace)});

    std::vector<double> plus_interior = base_interior,
                        minus_interior = base_interior;
    for (int i = 0; i < snapshot.interior_dofs; ++i)
    {
      plus_interior[i] += step * interior_direction[i];
      minus_interior[i] -= step * interior_direction[i];
    }
    std::vector<double> plus_r, plus_g, minus_r, minus_g;
    if (!AssembleIndependentQ(experiment, element, plus_interior, base_interior,
                              base_trace, plus_r, plus_g) ||
        !AssembleIndependentQ(experiment, element, minus_interior, base_interior,
                              base_trace, minus_r, minus_g)) return 7;
    std::vector<double> difference_r(snapshot.interior_dofs),
                        difference_g(snapshot.trace_dofs);
    for (int row = 0; row < snapshot.interior_dofs; ++row)
      difference_r[row] = (plus_r[row] - minus_r[row]) / (2.0 * step);
    for (int row = 0; row < snapshot.trace_dofs; ++row)
      difference_g[row] = (plus_g[row] - minus_g[row]) / (2.0 * step);
    const std::vector<double> a_action = ApplySnapshotBlock(
        snapshot.a, element, snapshot.interior_dofs, snapshot.interior_dofs,
        interior_direction);
    const std::vector<double> c_action = ApplySnapshotBlock(
        snapshot.c, element, snapshot.trace_dofs, snapshot.interior_dofs,
        interior_direction);
    const double interior_action_scale = std::max({1.0, InfinityNorm(a_action),
        InfinityNorm(c_action), InfinityNorm(difference_r), InfinityNorm(difference_g)});
    const double interior_tolerance = 2.0e-6 * interior_action_scale +
        256.0 * epsilon * q_scale / step;
    if (DifferenceNorm(difference_r, a_action) > interior_tolerance ||
        DifferenceNorm(difference_g, c_action) > interior_tolerance) return 8;

    std::vector<double> plus_trace = base_trace, minus_trace = base_trace;
    for (int i = 0; i < snapshot.trace_dofs; ++i)
    {
      plus_trace[i] += step * trace_direction[i];
      minus_trace[i] -= step * trace_direction[i];
    }
    if (!AssembleIndependentQ(experiment, element, base_interior, base_interior,
                              plus_trace, plus_r, plus_g) ||
        !AssembleIndependentQ(experiment, element, base_interior, base_interior,
                              minus_trace, minus_r, minus_g)) return 9;
    for (int row = 0; row < snapshot.interior_dofs; ++row)
      difference_r[row] = (plus_r[row] - minus_r[row]) / (2.0 * step);
    for (int row = 0; row < snapshot.trace_dofs; ++row)
      difference_g[row] = (plus_g[row] - minus_g[row]) / (2.0 * step);
    const std::vector<double> b_action = ApplySnapshotBlock(
        snapshot.b, element, snapshot.interior_dofs, snapshot.trace_dofs,
        trace_direction);
    const std::vector<double> d_action = ApplySnapshotBlock(
        snapshot.d, element, snapshot.trace_dofs, snapshot.trace_dofs,
        trace_direction);
    const double trace_action_scale = std::max({1.0, InfinityNorm(b_action),
        InfinityNorm(d_action), InfinityNorm(difference_r), InfinityNorm(difference_g)});
    const double trace_tolerance = 2.0e-6 * trace_action_scale +
        256.0 * epsilon * q_scale / step;
    if (DifferenceNorm(difference_r, b_action) > trace_tolerance ||
        DifferenceNorm(difference_g, d_action) > trace_tolerance) return 10;
  }
  return 0;
}

int TestTwoSidedP2OrientationOracle()
{
  SmoothEulerExperiment experiment(2);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage() ||
      !experiment.AssembleLinearization()) return 1;
  LocalLinearizationSnapshot snapshot;
  if (!experiment.CopyLocalLinearization(snapshot) ||
      !SnapshotHasExactExtents(snapshot)) return 2;
  OrientationCase orientation;
  if (!FindTwoSidedOrientationCase(experiment, orientation)) return 3;
  const mfem::FiniteElementSpace* trace_space = experiment.trace_space();
  if (trace_space == nullptr || trace_space->GetMesh() == nullptr) return 4;
  constexpr double epsilon = experiment_epsilon;
  double elem2_correct_error = 0.0;
  double elem2_wrong_normal_error = 0.0;
  double elem2_wrong_permutation_error = 0.0;
  double reversal_gap = std::numeric_limits<double>::infinity();
  int elem1_evaluations = 0;
  int elem2_evaluations = 0;
  auto report_orientation_controls = [&]()
  {
    const double ratio_floor =
        100.0 * std::max(elem2_correct_error, epsilon);
    std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10)
              << "DETAIL predicate=two_sided_p2_orientation_controls"
              << " correct_error=" << elem2_correct_error
              << " wrong_normal_error=" << elem2_wrong_normal_error
              << " wrong_permutation_error=" << elem2_wrong_permutation_error
              << " reversal_gap=" << reversal_gap
              << " elem1_evaluations=" << elem1_evaluations
              << " elem2_evaluations=" << elem2_evaluations
              << " absolute_threshold=1e-5"
              << " ratio_floor=" << ratio_floor
              << " reversal_threshold=0.5\n";
  };
  const int elements[2] = {orientation.element_one, orientation.element_two};
  for (const int element : elements)
  {
    if (element == orientation.element_one) ++elem1_evaluations;
    else if (element == orientation.element_two) ++elem2_evaluations;
    else
    {
      report_orientation_controls();
      return 15;
    }
    std::vector<double> base_interior, base_trace;
    if (!ExtractElementCoefficients(experiment, element, base_interior, base_trace))
      return 5;
    const int trace_offset = FindElementFaceOffset(
        *trace_space->GetMesh(), *trace_space, element, orientation.face);
    const mfem::FiniteElement* face_fe = trace_space->GetFaceElement(orientation.face);
    if (trace_offset < 0 || face_fe == nullptr || face_fe->GetDof() != 3) return 6;
    std::vector<double> interior_direction(snapshot.interior_dofs),
                        trace_direction(snapshot.trace_dofs, 0.0);
    for (int i = 0; i < snapshot.interior_dofs; ++i)
      interior_direction[i] = std::sin(0.41 * (i + 1)) -
                              0.27 * std::cos(0.13 * (i + 4));
    NormalizeInfinity(interior_direction);
    const double offsets[3] = {-0.07, 0.04, 0.19};
    // These non-palindromic values define only the normalized action w; the
    // captured production base trace remains unchanged in base_trace.
    for (int component = 0; component < conserved_components; ++component)
      for (int node = 0; node < 3; ++node)
      {
        trace_direction[trace_offset + node + 3 * component] =
            offsets[node] * (1.0 + 0.25 * component);
      }
    NormalizeInfinity(trace_direction);
    double local_reversal_gap = 0.0;
    for (int component = 0; component < conserved_components; ++component)
      for (int node = 0; node < 3; ++node)
      {
        local_reversal_gap = std::max(local_reversal_gap, std::abs(
            trace_direction[trace_offset + node + 3 * component] -
            trace_direction[trace_offset + (2 - node) + 3 * component]));
      }
    reversal_gap = std::min(reversal_gap, local_reversal_gap);
    if (trace_direction[trace_offset] == trace_direction[trace_offset + 1] ||
        trace_direction[trace_offset + 1] == trace_direction[trace_offset + 2] ||
        trace_direction[trace_offset] == trace_direction[trace_offset + 2] ||
        InfinityNorm(trace_direction) != 1.0)
    {
      report_orientation_controls();
      return 7;
    }
    const double step = std::cbrt(epsilon) *
        std::max({1.0, InfinityNorm(base_interior), InfinityNorm(base_trace)});

    std::vector<double> residual_r, residual_g;
    if (!AssembleIndependentQ(experiment, element, base_interior, base_interior,
                              base_trace, residual_r, residual_g, &orientation)) return 8;
    std::vector<double> expected_r(snapshot.interior_dofs),
                        expected_g(snapshot.trace_dofs);
    for (int row = 0; row < snapshot.interior_dofs; ++row)
      expected_r[row] = -snapshot.f[static_cast<std::size_t>(element) *
                                     snapshot.interior_dofs + row];
    for (int row = 0; row < snapshot.trace_dofs; ++row)
      expected_g[row] = -snapshot.g[static_cast<std::size_t>(element) *
                                     snapshot.trace_dofs + row];
    const double q_scale = std::max({1.0, InfinityNorm(residual_r),
        InfinityNorm(residual_g)});
    if (DifferenceNorm(residual_r, expected_r) > 2048.0 * epsilon * q_scale ||
        DifferenceNorm(residual_g, expected_g) > 2048.0 * epsilon * q_scale)
      return 9;

    std::vector<double> plus_interior = base_interior,
                        minus_interior = base_interior;
    for (int i = 0; i < snapshot.interior_dofs; ++i)
    {
      plus_interior[i] += step * interior_direction[i];
      minus_interior[i] -= step * interior_direction[i];
    }
    std::vector<double> plus_r, plus_g, minus_r, minus_g;
    if (!AssembleIndependentQ(experiment, element, plus_interior, base_interior,
                              base_trace, plus_r, plus_g, &orientation) ||
        !AssembleIndependentQ(experiment, element, minus_interior, base_interior,
                              base_trace, minus_r, minus_g, &orientation)) return 10;
    std::vector<double> difference_r(snapshot.interior_dofs),
                        difference_g(snapshot.trace_dofs);
    for (int row = 0; row < snapshot.interior_dofs; ++row)
      difference_r[row] = (plus_r[row] - minus_r[row]) / (2.0 * step);
    for (int row = 0; row < snapshot.trace_dofs; ++row)
      difference_g[row] = (plus_g[row] - minus_g[row]) / (2.0 * step);
    const std::vector<double> a_action = ApplySnapshotBlock(
        snapshot.a, element, snapshot.interior_dofs, snapshot.interior_dofs,
        interior_direction);
    const std::vector<double> c_action = ApplySnapshotBlock(
        snapshot.c, element, snapshot.trace_dofs, snapshot.interior_dofs,
        interior_direction);
    const double interior_scale = std::max({1.0, InfinityNorm(a_action),
        InfinityNorm(c_action), InfinityNorm(difference_r), InfinityNorm(difference_g)});
    const double interior_tolerance = 2.0e-6 * interior_scale +
        256.0 * epsilon * q_scale / step;
    if (DifferenceNorm(difference_r, a_action) > interior_tolerance ||
        DifferenceNorm(difference_g, c_action) > interior_tolerance) return 11;

    const std::vector<double> b_action = ApplySnapshotBlock(
        snapshot.b, element, snapshot.interior_dofs, snapshot.trace_dofs,
        trace_direction);
    const std::vector<double> d_action = ApplySnapshotBlock(
        snapshot.d, element, snapshot.trace_dofs, snapshot.trace_dofs,
        trace_direction);
    auto trace_action_error = [&](const OrientationMutation mutation,
                                  std::vector<double>& action_r,
                                  std::vector<double>& action_g) -> double
    {
      std::vector<double> plus_trace = base_trace, minus_trace = base_trace;
      for (int i = 0; i < snapshot.trace_dofs; ++i)
      {
        plus_trace[i] += step * trace_direction[i];
        minus_trace[i] -= step * trace_direction[i];
      }
      if (!AssembleIndependentQ(experiment, element, base_interior, base_interior,
                                plus_trace, plus_r, plus_g, &orientation, mutation) ||
          !AssembleIndependentQ(experiment, element, base_interior, base_interior,
                                minus_trace, minus_r, minus_g, &orientation, mutation))
      {
        return std::numeric_limits<double>::infinity();
      }
      action_r.resize(snapshot.interior_dofs);
      action_g.resize(snapshot.trace_dofs);
      for (int row = 0; row < snapshot.interior_dofs; ++row)
        action_r[row] = (plus_r[row] - minus_r[row]) / (2.0 * step);
      for (int row = 0; row < snapshot.trace_dofs; ++row)
        action_g[row] = (plus_g[row] - minus_g[row]) / (2.0 * step);
      return NormalizedTraceActionError(action_r, action_g, b_action, d_action);
    };
    std::vector<double> correct_r, correct_g;
    const double correct_error = trace_action_error(
        OrientationMutation::none, correct_r, correct_g);
    const double trace_scale = std::max({1.0, InfinityNorm(b_action),
        InfinityNorm(d_action), InfinityNorm(correct_r), InfinityNorm(correct_g)});
    const double trace_tolerance = 2.0e-6 * trace_scale +
        256.0 * epsilon * q_scale / step;
    if (element == orientation.element_two) elem2_correct_error = correct_error;
    if (!std::isfinite(correct_error) ||
        DifferenceNorm(correct_r, b_action) > trace_tolerance ||
        DifferenceNorm(correct_g, d_action) > trace_tolerance)
    {
      report_orientation_controls();
      return 12;
    }
    if (element == orientation.element_two)
    {
      std::vector<double> mutated_r, mutated_g;
      elem2_wrong_normal_error = trace_action_error(
          OrientationMutation::wrong_elem2_normal, mutated_r, mutated_g);
      elem2_wrong_permutation_error = trace_action_error(
          OrientationMutation::wrong_elem2_permutation, mutated_r, mutated_g);
    }
  }
  if (elem1_evaluations != 1 || elem2_evaluations != 1)
  {
    report_orientation_controls();
    return 15;
  }
  if (!(reversal_gap > 0.5))
  {
    report_orientation_controls();
    return 7;
  }
  const double ratio_floor = 100.0 * std::max(elem2_correct_error, epsilon);
  if (!std::isfinite(elem2_wrong_normal_error) ||
      elem2_wrong_normal_error < 1.0e-5 ||
      elem2_wrong_normal_error < ratio_floor)
  {
    report_orientation_controls();
    return 13;
  }
  if (!std::isfinite(elem2_wrong_permutation_error) ||
      elem2_wrong_permutation_error < 1.0e-5 ||
      elem2_wrong_permutation_error < ratio_floor)
  {
    report_orientation_controls();
    return 14;
  }
  return 0;
}

bool SolveDense(std::vector<double> matrix, std::vector<double>& rhs, const int n)
{
  if (static_cast<int>(matrix.size()) != n * n || static_cast<int>(rhs.size()) != n)
  {
    return false;
  }
  for (int pivot = 0; pivot < n; ++pivot)
  {
    int best = pivot;
    for (int row = pivot + 1; row < n; ++row)
    {
      if (std::abs(matrix[static_cast<std::size_t>(row) * n + pivot]) >
          std::abs(matrix[static_cast<std::size_t>(best) * n + pivot])) best = row;
    }
    if (std::abs(matrix[static_cast<std::size_t>(best) * n + pivot]) < 1.0e-13)
    {
      return false;
    }
    for (int column = pivot; column < n; ++column)
    {
      std::swap(matrix[static_cast<std::size_t>(pivot) * n + column],
                matrix[static_cast<std::size_t>(best) * n + column]);
    }
    std::swap(rhs[pivot], rhs[best]);
    const double diagonal = matrix[static_cast<std::size_t>(pivot) * n + pivot];
    for (int column = pivot; column < n; ++column)
    {
      matrix[static_cast<std::size_t>(pivot) * n + column] /= diagonal;
    }
    rhs[pivot] /= diagonal;
    for (int row = 0; row < n; ++row)
    {
      if (row == pivot) { continue; }
      const double factor = matrix[static_cast<std::size_t>(row) * n + pivot];
      for (int column = pivot; column < n; ++column)
      {
        matrix[static_cast<std::size_t>(row) * n + column] -=
            factor * matrix[static_cast<std::size_t>(pivot) * n + column];
      }
      rhs[row] -= factor * rhs[pivot];
    }
  }
  return true;
}

std::vector<double> IndependentlyCondense(const LocalLinearizationSnapshot& snapshot,
                                          const int element)
{
  const int ni = snapshot.interior_dofs;
  const int nt = snapshot.trace_dofs;
  const std::size_t a_offset = static_cast<std::size_t>(element) * ni * ni;
  const std::size_t b_offset = static_cast<std::size_t>(element) * ni * nt;
  const std::size_t c_offset = static_cast<std::size_t>(element) * nt * ni;
  const std::size_t d_offset = static_cast<std::size_t>(element) * nt * nt;
  std::vector<double> result(nt * nt);
  for (int row = 0; row < nt; ++row)
    for (int column = 0; column < nt; ++column)
    {
      std::vector<double> rhs(ni);
      for (int interior = 0; interior < ni; ++interior)
      {
        rhs[interior] = snapshot.b[b_offset + static_cast<std::size_t>(interior) * nt + column];
      }
      std::vector<double> a(snapshot.a.begin() + a_offset,
                            snapshot.a.begin() + a_offset + ni * ni);
      if (!SolveDense(std::move(a), rhs, ni)) { return {}; }
      double reduced = snapshot.d[d_offset + static_cast<std::size_t>(row) * nt + column];
      for (int interior = 0; interior < ni; ++interior)
      {
        reduced -= snapshot.c[c_offset + static_cast<std::size_t>(row) * ni + interior] *
                   rhs[interior];
      }
      result[static_cast<std::size_t>(row) * nt + column] = reduced;
    }
  return result;
}

std::vector<double> ResolveInteriorThenEvaluateTrace(
    const LocalLinearizationSnapshot& snapshot, const int element,
    const std::vector<double>& trace)
{
  const int ni = snapshot.interior_dofs;
  const int nt = snapshot.trace_dofs;
  if (static_cast<int>(trace.size()) != nt) return {};
  const std::size_t a_offset = static_cast<std::size_t>(element) * ni * ni;
  const std::size_t b_offset = static_cast<std::size_t>(element) * ni * nt;
  const std::size_t c_offset = static_cast<std::size_t>(element) * nt * ni;
  const std::size_t d_offset = static_cast<std::size_t>(element) * nt * nt;
  std::vector<double> rhs(ni);
  for (int row = 0; row < ni; ++row)
  {
    rhs[row] = snapshot.f[static_cast<std::size_t>(element) * ni + row];
    for (int column = 0; column < nt; ++column)
    {
      rhs[row] -= snapshot.b[b_offset + static_cast<std::size_t>(row) * nt + column] *
                  trace[column];
    }
  }
  std::vector<double> a(snapshot.a.begin() + a_offset,
                        snapshot.a.begin() + a_offset + ni * ni);
  if (!SolveDense(std::move(a), rhs, ni)) return {};
  std::vector<double> residual(nt);
  for (int row = 0; row < nt; ++row)
  {
    residual[row] = -snapshot.g[static_cast<std::size_t>(element) * nt + row];
    for (int column = 0; column < ni; ++column)
    {
      residual[row] += snapshot.c[c_offset + static_cast<std::size_t>(row) * ni + column] *
                       rhs[column];
    }
    for (int column = 0; column < nt; ++column)
    {
      residual[row] += snapshot.d[d_offset + static_cast<std::size_t>(row) * nt + column] *
                       trace[column];
    }
  }
  return residual;
}

int TestFixedScalarTauConstantAndConservationOracles()
{
  const ConservedState state{{1.1, 0.3, -0.2, 2.9}};
  const std::array<double, 2> normal{{0.6, -0.8}};
  FluxVector positive{}, negative{};
  if (!NormalFlux(state, normal, positive).ok() ||
      !NormalFlux(state, {{-normal[0], -normal[1]}}, negative).ok()) return 1;
  // With U=Uhat the tau term is independently zero, and the two oriented
  // numerical fluxes must cancel component-wise.
  for (int component = 0; component < conserved_components; ++component)
  {
    if (!Close(positive[component] + negative[component], 0.0, 3.0e-14)) return 2;
  }
  const ConservedState left{{1.0, 0.2, -0.1, 2.6}};
  const ConservedState right{{1.2, 0.1, -0.3, 3.0}};
  const ConservedState trace{{1.1, 0.15, -0.2, 2.8}};
  FluxVector trace_positive{}, trace_negative{};
  if (!NormalFlux(trace, normal, trace_positive).ok() ||
      !NormalFlux(trace, {{-normal[0], -normal[1]}}, trace_negative).ok()) return 3;
  // This is separate skeleton conservation accounting: advective terms cancel
  // by orientation and the remaining jump is exactly the two tau penalties.
  for (int component = 0; component < conserved_components; ++component)
  {
    const double accounted = trace_positive[component] + trace_negative[component] +
        fixed_scalar_tau * (left[component] - trace[component]) +
        fixed_scalar_tau * (right[component] - trace[component]);
    const double expected = fixed_scalar_tau *
        (left[component] + right[component] - 2.0 * trace[component]);
    if (!Close(accounted, expected, 3.0e-14)) return 4;
  }
  return 0;
}

int TestFaceBlockDirectionalAndIndependentSchurOracles()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage() ||
      !experiment.AssembleLinearization()) return 1;
  LocalLinearizationSnapshot snapshot;
  if (!experiment.CopyLocalLinearization(snapshot) || snapshot.elements <= 0 ||
      snapshot.a.empty() || snapshot.b.empty() || snapshot.c.empty() || snapshot.d.empty())
  {
    return 2;
  }
  // Independent directional finite differences of the two arguments of
  // F(Uhat).n + tau(U-Uhat), which respectively exercise A/C and B/D signs.
  const ConservedState interior{{1.15, 0.31, -0.18, 2.95}};
  const ConservedState trace{{1.08, 0.22, -0.27, 2.85}};
  const ConservedState direction{{0.03, -0.02, 0.04, 0.05}};
  const std::array<double, 2> normal{{0.8, 0.6}};
  FluxJacobian jacobian{};
  if (!NormalFluxJacobian(trace, normal, jacobian).ok()) return 3;
  constexpr double h = 1.0e-7;
  for (int varied_trace = 0; varied_trace < 2; ++varied_trace)
  {
    ConservedState plus_interior = interior, minus_interior = interior;
    ConservedState plus_trace = trace, minus_trace = trace;
    for (int component = 0; component < conserved_components; ++component)
    {
      (varied_trace ? plus_trace : plus_interior)[component] += h * direction[component];
      (varied_trace ? minus_trace : minus_interior)[component] -= h * direction[component];
    }
    FluxVector fp{}, fm{};
    if (!NormalFlux(plus_trace, normal, fp).ok() || !NormalFlux(minus_trace, normal, fm).ok())
    {
      return 4;
    }
    for (int row = 0; row < conserved_components; ++row)
    {
      const double difference = (fp[row] - fm[row]) / (2.0 * h) +
          fixed_scalar_tau * ((plus_interior[row] - plus_trace[row]) -
                              (minus_interior[row] - minus_trace[row])) / (2.0 * h);
      double analytic = 0.0;
      for (int column = 0; column < conserved_components; ++column)
      {
        analytic += (varied_trace ? jacobian[row][column] -
                                      (row == column ? fixed_scalar_tau : 0.0) :
                                    (row == column ? fixed_scalar_tau : 0.0)) * direction[column];
      }
      if (!Close(difference, analytic, 3.0e-6)) return 5;
    }
  }
  if (!experiment.CondenseAndAssembleTrace() || experiment.trace_matrix() == nullptr) return 6;
  const mfem::SparseMatrix& condensed = *experiment.trace_matrix();
  std::vector<double> monolithic(static_cast<std::size_t>(condensed.Height()) * condensed.Width());
  for (int element = 0; element < snapshot.elements; ++element)
  {
    const std::vector<double> local = IndependentlyCondense(snapshot, element);
    if (local.empty()) return 7;
    for (int row = 0; row < snapshot.trace_dofs; ++row)
    {
      double row_sign = 1.0;
      const int global_row = DecodeSignedVDof(
          snapshot.signed_trace_dofs[static_cast<std::size_t>(element) * snapshot.trace_dofs + row], row_sign);
      for (int column = 0; column < snapshot.trace_dofs; ++column)
      {
        double column_sign = 1.0;
        const int global_column = DecodeSignedVDof(
            snapshot.signed_trace_dofs[static_cast<std::size_t>(element) * snapshot.trace_dofs + column], column_sign);
        monolithic[static_cast<std::size_t>(global_row) * condensed.Width() + global_column] +=
            row_sign * column_sign * local[static_cast<std::size_t>(row) * snapshot.trace_dofs + column];
      }
    }
  }
  for (int row = 0; row < condensed.Height(); ++row)
    for (int column = 0; column < condensed.Width(); ++column)
    {
      if (!Close(monolithic[static_cast<std::size_t>(row) * condensed.Width() + column],
                 condensed(row, column), 2.0e-10)) return 8;
    }
  // Exercise the actual similarity-permuted BlockILU/GMRES path, then check
  // its global residual and an independently pivoted dense reference solve.
  if (!experiment.SolveTrace()) return 9;
  std::vector<double> trace_increment, trace_rhs;
  if (!experiment.CopyTraceIncrement(trace_increment) ||
      !experiment.CopyTraceRightHandSide(trace_rhs)) return 10;
  for (int row = 0; row < condensed.Height(); ++row)
  {
    double action = 0.0;
    for (int column = 0; column < condensed.Width(); ++column)
    {
      action += condensed(row, column) * trace_increment[column];
    }
    if (!Close(action, trace_rhs[row], 2.0e-9)) return 11;
  }
  std::vector<double> dense_matrix(static_cast<std::size_t>(condensed.Height()) * condensed.Width());
  for (int row = 0; row < condensed.Height(); ++row)
    for (int column = 0; column < condensed.Width(); ++column)
    {
      dense_matrix[static_cast<std::size_t>(row) * condensed.Width() + column] =
          condensed(row, column);
    }
  std::vector<double> dense_solution = trace_rhs;
  if (!SolveDense(std::move(dense_matrix), dense_solution, condensed.Height())) return 12;
  for (int row = 0; row < condensed.Height(); ++row)
  {
    if (!Close(trace_increment[row], dense_solution[row], 3.0e-8)) return 13;
  }
  // Independently resolve the local interior state for two perturbed traces,
  // then difference the reduced residual.  This is not ApplySchur applied to
  // itself: the interior solve and residual evaluation are reconstructed here.
  const std::vector<double> local_schur = IndependentlyCondense(snapshot, 0);
  std::vector<double> lambda(snapshot.trace_dofs, 0.0);
  for (int i = 0; i < snapshot.trace_dofs; ++i) { lambda[i] = 0.01 * (i + 1); }
  std::vector<double> plus = lambda, minus = lambda, expected(snapshot.trace_dofs, 0.0);
  constexpr double reduced_step = 1.0e-7;
  for (int i = 0; i < snapshot.trace_dofs; ++i)
  {
    plus[i] += reduced_step * lambda[i];
    minus[i] -= reduced_step * lambda[i];
  }
  const std::vector<double> reduced_plus = ResolveInteriorThenEvaluateTrace(snapshot, 0, plus);
  const std::vector<double> reduced_minus = ResolveInteriorThenEvaluateTrace(snapshot, 0, minus);
  if (reduced_plus.empty() || reduced_minus.empty()) return 14;
  for (int row = 0; row < snapshot.trace_dofs; ++row)
    for (int column = 0; column < snapshot.trace_dofs; ++column)
    {
      expected[row] += local_schur[static_cast<std::size_t>(row) * snapshot.trace_dofs + column] * lambda[column];
    }
  for (int row = 0; row < snapshot.trace_dofs; ++row)
  {
    const double derivative = (reduced_plus[row] - reduced_minus[row]) / (2.0 * reduced_step);
    if (!Close(derivative, expected[row], 2.0e-6)) return 15;
  }
  if (!experiment.RecoverInterior()) return 16;
  return 0;
}

struct IndependentContext
{
  std::unique_ptr<mfem::Mesh> mesh;
  std::unique_ptr<mfem::L2_FECollection> interior_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::FiniteElementSpace> interior_space;
  std::unique_ptr<mfem::FiniteElementSpace> trace_space;
  int polynomial_order = 0;
  int width = 0;
};

bool BuildIndependentContext(const int polynomial_order, const int width,
                             IndependentContext& context)
{
  context = {};
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        width, width, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations;
    translations.emplace_back(dimension);
    translations.emplace_back(dimension);
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    context.mesh = std::make_unique<mfem::Mesh>(mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations)));
    context.interior_collection = std::make_unique<mfem::L2_FECollection>(
        polynomial_order, dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    context.trace_collection =
        std::make_unique<mfem::DG_Interface_FECollection>(
            polynomial_order, dimension, mfem::FiniteElement::VALUE,
            mfem::BasisType::GaussLegendre);
    context.interior_space = std::make_unique<mfem::FiniteElementSpace>(
        context.mesh.get(), context.interior_collection.get(),
        conserved_components, mfem::Ordering::byNODES);
    context.trace_space = std::make_unique<mfem::FiniteElementSpace>(
        context.mesh.get(), context.trace_collection.get(), conserved_components,
        mfem::Ordering::byNODES);
  }
  catch (const std::exception&)
  {
    return false;
  }
  context.polynomial_order = polynomial_order;
  context.width = width;
  return context.mesh && context.interior_space && context.trace_space &&
         context.mesh->GetNE() == width * width &&
         context.mesh->GetNumFaces() == 2 * width * width;
}

bool OracleExactConservative(const double x, const double y,
                             ConservedState& state)
{
  constexpr double oracle_pi = 3.141592653589793238462643383279502884;
  const double rho = 1.0 + 0.1 * std::sin(2.0 * oracle_pi * x) *
                               std::sin(2.0 * oracle_pi * y);
  const double u = 0.35 + 0.05 * std::cos(2.0 * oracle_pi * x) *
                            std::sin(2.0 * oracle_pi * y);
  const double v = -0.20 + 0.05 * std::sin(2.0 * oracle_pi * x) *
                             std::cos(2.0 * oracle_pi * y);
  const double p = 1.0 + 0.1 * std::cos(2.0 * oracle_pi * x) *
                             std::cos(2.0 * oracle_pi * y);
  state = {{rho, rho * u, rho * v,
            p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)}};
  return std::all_of(state.begin(), state.end(),
                     [](const double value) { return std::isfinite(value); });
}

bool OraclePerturbedConservative(const double x, const double y,
                                 ConservedState& state)
{
  constexpr double oracle_pi = 3.141592653589793238462643383279502884;
  const double psi_one = std::sin(2.0 * oracle_pi * x) *
                         std::cos(2.0 * oracle_pi * y);
  const double psi_two = std::cos(2.0 * oracle_pi * x) *
                         std::sin(2.0 * oracle_pi * y);
  const double psi_three = std::sin(2.0 * oracle_pi * (x + y));
  const double rho_exact = 1.0 + 0.1 * std::sin(2.0 * oracle_pi * x) *
                                     std::sin(2.0 * oracle_pi * y);
  const double u_exact = 0.35 + 0.05 * std::cos(2.0 * oracle_pi * x) *
                                  std::sin(2.0 * oracle_pi * y);
  const double v_exact = -0.20 + 0.05 * std::sin(2.0 * oracle_pi * x) *
                                   std::cos(2.0 * oracle_pi * y);
  const double p_exact = 1.0 + 0.1 * std::cos(2.0 * oracle_pi * x) *
                                   std::cos(2.0 * oracle_pi * y);
  const double rho = rho_exact * (1.0 + 0.04 * psi_one);
  const double u = u_exact + 0.03 * psi_two;
  const double v = v_exact - 0.03 * psi_one;
  const double p = p_exact * (1.0 + 0.04 * psi_three);
  state = {{rho, rho * u, rho * v,
            p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)}};
  return std::all_of(state.begin(), state.end(),
                     [](const double value) { return std::isfinite(value); });
}

bool IndependentProjection(const IndependentContext& context,
                           const bool perturbed,
                           std::vector<double>& interior,
                           std::vector<double>& trace)
{
  auto coefficient = [perturbed](const mfem::Vector& point, mfem::Vector& value)
  {
    ConservedState state{};
    const bool okay = perturbed ?
        OraclePerturbedConservative(point[0], point[1], state) :
        OracleExactConservative(point[0], point[1], state);
    if (!okay) throw std::runtime_error("independent coefficient failed");
    value.SetSize(conserved_components);
    for (int component = 0; component < conserved_components; ++component)
    {
      value[component] = state[component];
    }
  };
  try
  {
    mfem::VectorFunctionCoefficient exact(conserved_components, coefficient);
    mfem::GridFunction interior_field(context.interior_space.get());
    mfem::GridFunction trace_field(context.trace_space.get());
    interior_field.ProjectCoefficient(exact);
    trace_field = 0.0;
    for (int face = 0; face < context.mesh->GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* fe = context.trace_space->GetFaceElement(face);
      mfem::ElementTransformation* transformation =
          context.mesh->GetFaceTransformation(face);
      if (fe == nullptr || transformation == nullptr) return false;
      mfem::Array<int> vdofs;
      context.trace_space->GetFaceVDofs(face, vdofs);
      const mfem::IntegrationRule& nodes = fe->GetNodes();
      for (int node = 0; node < fe->GetDof(); ++node)
      {
        mfem::Vector physical(dimension), value(conserved_components);
        transformation->Transform(nodes.IntPoint(node), physical);
        coefficient(physical, value);
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = node + fe->GetDof() * component;
          double sign = 1.0;
          const int global = OracleDecodeSignedVDof(vdofs[local], sign);
          if (global < 0 || global >= trace_field.Size()) return false;
          trace_field[global] = sign * value[component];
        }
      }
    }
    interior.assign(interior_field.GetData(),
                    interior_field.GetData() + interior_field.Size());
    trace.assign(trace_field.GetData(), trace_field.GetData() + trace_field.Size());
  }
  catch (const std::exception&)
  {
    return false;
  }
  return true;
}

bool ExtractIndependentElement(const IndependentContext& context,
                               const std::vector<double>& global_interior,
                               const std::vector<double>& global_trace,
                               const int element,
                               std::vector<double>& local_interior,
                               std::vector<double>& local_trace,
                               std::vector<int>* signed_trace_dofs = nullptr)
{
  mfem::Array<int> interior_vdofs;
  context.interior_space->GetElementVDofs(element, interior_vdofs);
  local_interior.resize(interior_vdofs.Size());
  for (int local = 0; local < interior_vdofs.Size(); ++local)
  {
    double sign = 1.0;
    const int global = OracleDecodeSignedVDof(interior_vdofs[local], sign);
    if (global < 0 || global >= static_cast<int>(global_interior.size())) return false;
    local_interior[local] = sign * global_interior[global];
  }
  local_trace.clear();
  if (signed_trace_dofs != nullptr) signed_trace_dofs->clear();
  mfem::Array<int> faces, orientations;
  context.mesh->GetElementEdges(element, faces, orientations);
  for (int local_face = 0; local_face < faces.Size(); ++local_face)
  {
    mfem::Array<int> vdofs;
    context.trace_space->GetFaceVDofs(faces[local_face], vdofs);
    for (int local = 0; local < vdofs.Size(); ++local)
    {
      double sign = 1.0;
      const int global = OracleDecodeSignedVDof(vdofs[local], sign);
      if (global < 0 || global >= static_cast<int>(global_trace.size())) return false;
      local_trace.push_back(sign * global_trace[global]);
      if (signed_trace_dofs != nullptr) signed_trace_dofs->push_back(vdofs[local]);
    }
  }
  return true;
}

bool AssembleIndependentGlobalQ(
    const IndependentContext& context, const int quadrature_order,
    const std::vector<double>& old_interior,
    const std::vector<double>& current_interior,
    const std::vector<double>& current_trace,
    std::vector<double>& residual_interior,
    std::vector<double>& residual_trace,
    std::vector<int>& signed_trace_dofs)
{
  residual_interior.clear();
  residual_trace.clear();
  signed_trace_dofs.clear();
  for (int element = 0; element < context.mesh->GetNE(); ++element)
  {
    std::vector<double> old_local, current_local, trace_local, ignored_trace,
        local_r, local_g;
    std::vector<int> map;
    if (!ExtractIndependentElement(context, old_interior, current_trace, element,
                                   old_local, ignored_trace) ||
        !ExtractIndependentElement(context, current_interior, current_trace,
                                   element, current_local, trace_local, &map) ||
        !AssembleIndependentQOnSpaces(
            *context.interior_space, *context.trace_space, quadrature_order,
            element, current_local, old_local, trace_local, local_r, local_g))
    {
      return false;
    }
    residual_interior.insert(residual_interior.end(), local_r.begin(), local_r.end());
    residual_trace.insert(residual_trace.end(), local_g.begin(), local_g.end());
    signed_trace_dofs.insert(signed_trace_dofs.end(), map.begin(), map.end());
  }
  return true;
}

struct OracleAdmissibility
{
  bool passed = false;
  TrialRejectionReason reason = TrialRejectionReason::None;
  double minimum_density = std::numeric_limits<double>::infinity();
  double minimum_pressure = std::numeric_limits<double>::infinity();
  double maximum_wave_speed = 0.0;
};

void OracleObserveState(const ConservedState& state, bool& nonfinite,
                        bool& density, bool& pressure, bool& wave,
                        OracleAdmissibility& result)
{
  if (!std::all_of(state.begin(), state.end(),
                   [](const double value) { return std::isfinite(value); }))
  {
    nonfinite = true;
    return;
  }
  result.minimum_density = std::min(result.minimum_density, state[0]);
  if (!(state[0] >= 1.0e-8))
  {
    density = true;
    return;
  }
  const double p = (gamma - 1.0) *
      (state[3] - 0.5 * (state[1] * state[1] + state[2] * state[2]) /
                      state[0]);
  if (!std::isfinite(p))
  {
    nonfinite = true;
    return;
  }
  result.minimum_pressure = std::min(result.minimum_pressure, p);
  if (!(p >= 1.0e-8))
  {
    pressure = true;
    return;
  }
  const double rho = state[0];
  const double u = state[1] / rho;
  const double v = state[2] / rho;
  const double speed = std::sqrt(u * u + v * v) +
                       std::sqrt(gamma * p / rho);
  if (!std::isfinite(speed))
  {
    nonfinite = true;
    return;
  }
  result.maximum_wave_speed = std::max(result.maximum_wave_speed, speed);
  if (!(speed < fixed_scalar_tau)) wave = true;
}

bool IndependentPrescreen(const IndependentContext& context,
                          const int quadrature_order,
                          const std::vector<double>& global_interior,
                          const std::vector<double>& global_trace,
                          OracleAdmissibility& result)
{
  result = {};
  bool nonfinite = false, density = false, pressure = false, wave = false;
  for (int element = 0; element < context.mesh->GetNE(); ++element)
  {
    std::vector<double> local_interior, local_trace;
    if (!ExtractIndependentElement(context, global_interior, global_trace, element,
                                   local_interior, local_trace)) return false;
    const mfem::FiniteElement* fe = context.interior_space->GetFE(element);
    if (fe == nullptr) return false;
    mfem::Vector shape(fe->GetDof());
    const mfem::IntegrationRule& volume_rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    for (int q = 0; q < volume_rule.GetNPoints(); ++q)
    {
      fe->CalcShape(volume_rule.IntPoint(q), shape);
      ConservedState state{};
      if (!InterpolateOracleState(local_interior, fe->GetDof(), 0, shape, state))
      {
        nonfinite = true;
      }
      else
      {
        OracleObserveState(state, nonfinite, density, pressure, wave, result);
      }
    }
    mfem::Array<int> faces, orientations;
    context.mesh->GetElementEdges(element, faces, orientations);
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      mfem::FaceElementTransformations* transformation =
          context.mesh->GetFaceElementTransformations(faces[local_face]);
      const mfem::FiniteElement* trace_fe =
          context.trace_space->GetFaceElement(faces[local_face]);
      if (transformation == nullptr || trace_fe == nullptr) return false;
      const bool first = transformation->Elem1No == element;
      if (!first && transformation->Elem2No != element) return false;
      const mfem::IntegrationRule& rule =
          mfem::IntRules.Get(trace_fe->GetGeomType(), quadrature_order);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        transformation->SetAllIntPoints(&rule.IntPoint(q));
        const mfem::IntegrationPoint& element_point = first ?
            transformation->GetElement1IntPoint() :
            transformation->GetElement2IntPoint();
        fe->CalcShape(element_point, shape);
        ConservedState state{};
        if (!InterpolateOracleState(local_interior, fe->GetDof(), 0, shape, state))
        {
          nonfinite = true;
        }
        else
        {
          OracleObserveState(state, nonfinite, density, pressure, wave, result);
        }
      }
    }
  }
  for (int face = 0; face < context.mesh->GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* fe = context.trace_space->GetFaceElement(face);
    if (fe == nullptr) return false;
    mfem::Array<int> vdofs;
    context.trace_space->GetFaceVDofs(face, vdofs);
    std::vector<double> local_trace(vdofs.Size());
    for (int local = 0; local < vdofs.Size(); ++local)
    {
      double sign = 1.0;
      const int global = OracleDecodeSignedVDof(vdofs[local], sign);
      if (global < 0 || global >= static_cast<int>(global_trace.size())) return false;
      local_trace[local] = sign * global_trace[global];
    }
    mfem::Vector shape(fe->GetDof());
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      fe->CalcShape(rule.IntPoint(q), shape);
      ConservedState state{};
      if (!InterpolateOracleState(local_trace, fe->GetDof(), 0, shape, state))
      {
        nonfinite = true;
      }
      else
      {
        OracleObserveState(state, nonfinite, density, pressure, wave, result);
      }
    }
  }
  result.reason = nonfinite ? TrialRejectionReason::Nonfinite :
      density ? TrialRejectionReason::Density :
      pressure ? TrialRejectionReason::Pressure :
      wave ? TrialRejectionReason::WaveSpeed : TrialRejectionReason::None;
  result.passed = result.reason == TrialRejectionReason::None;
  return true;
}

struct ContinuousOracleResult
{
  ErrorIntegralEvidence volume{};
  ErrorIntegralEvidence trace{};
  double maximum_face_measure_error = 0.0;
};

bool IndependentContinuousErrors(const IndependentContext& context,
                                 const std::vector<double>& global_interior,
                                 const std::vector<double>& global_trace,
                                 const int quadrature_order,
                                 ContinuousOracleResult& result)
{
  result = {};
  long double volume_error = 0.0L, volume_norm = 0.0L, volume_measure = 0.0L;
  for (int element = 0; element < context.mesh->GetNE(); ++element)
  {
    std::vector<double> local_interior, local_trace;
    if (!ExtractIndependentElement(context, global_interior, global_trace, element,
                                   local_interior, local_trace)) return false;
    const mfem::FiniteElement* fe = context.interior_space->GetFE(element);
    mfem::ElementTransformation* transformation =
        context.mesh->GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr) return false;
    mfem::Vector shape(fe->GetDof());
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      mfem::Vector physical(dimension);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      ConservedState numerical{}, exact{};
      if (!InterpolateOracleState(local_interior, fe->GetDof(), 0, shape,
                                  numerical) ||
          !OracleExactConservative(physical[0], physical[1], exact)) return false;
      const double weight = point.weight * transformation->Weight();
      if (!(weight > 0.0) || !std::isfinite(weight)) return false;
      volume_measure += weight;
      for (int component = 0; component < conserved_components; ++component)
      {
        const long double difference = numerical[component] - exact[component];
        volume_error += weight * difference * difference;
        volume_norm += weight * static_cast<long double>(exact[component]) *
                       exact[component];
      }
    }
  }
  long double trace_error = 0.0L, trace_norm = 0.0L, trace_measure = 0.0L;
  const double expected_face_measure = 1.0 / context.width;
  for (int face = 0; face < context.mesh->GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* fe = context.trace_space->GetFaceElement(face);
    mfem::ElementTransformation* transformation =
        context.mesh->GetFaceTransformation(face);
    if (fe == nullptr || transformation == nullptr) return false;
    mfem::Array<int> vdofs;
    context.trace_space->GetFaceVDofs(face, vdofs);
    std::vector<double> local_trace(vdofs.Size());
    for (int local = 0; local < vdofs.Size(); ++local)
    {
      double sign = 1.0;
      const int global = OracleDecodeSignedVDof(vdofs[local], sign);
      if (global < 0 || global >= static_cast<int>(global_trace.size())) return false;
      local_trace[local] = sign * global_trace[global];
    }
    mfem::Vector shape(fe->GetDof());
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    long double face_measure = 0.0L;
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      mfem::Vector scaled_normal(dimension);
      mfem::CalcOrtho(transformation->Jacobian(), scaled_normal);
      const double face_jacobian = scaled_normal.Norml2();
      if (!(face_jacobian > 0.0) || !std::isfinite(face_jacobian)) return false;
      const std::array<double, dimension> normal{{
          scaled_normal[0] / face_jacobian,
          scaled_normal[1] / face_jacobian}};
      if (!std::isfinite(normal[0]) || !std::isfinite(normal[1])) return false;
      mfem::Vector physical(dimension);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      ConservedState numerical{}, exact{};
      if (!InterpolateOracleState(local_trace, fe->GetDof(), 0, shape,
                                  numerical) ||
          !OracleExactConservative(physical[0], physical[1], exact)) return false;
      const double weight = point.weight * face_jacobian;
      if (!(weight > 0.0) || !std::isfinite(weight)) return false;
      face_measure += weight;
      trace_measure += weight;
      for (int component = 0; component < conserved_components; ++component)
      {
        const long double difference = numerical[component] - exact[component];
        trace_error += weight * difference * difference;
        trace_norm += weight * static_cast<long double>(exact[component]) *
                      exact[component];
      }
    }
    result.maximum_face_measure_error = std::max(
        result.maximum_face_measure_error,
        std::abs(static_cast<double>(face_measure) - expected_face_measure));
  }
  auto publish = [](const long double error, const long double norm,
                    const long double measure, const int faces,
                    ErrorIntegralEvidence& output)
  {
    if (!(error > 0.0L) || !(norm > 0.0L) || !(measure > 0.0L)) return false;
    output.error_squared = static_cast<double>(error);
    output.exact_norm_squared = static_cast<double>(norm);
    output.measure = static_cast<double>(measure);
    output.direct_l2 = std::sqrt(output.error_squared);
    output.rms = std::sqrt(output.error_squared / output.measure);
    output.relative_l2 = std::sqrt(output.error_squared / output.exact_norm_squared);
    output.unique_face_count = faces;
    return std::isfinite(output.error_squared) &&
           std::isfinite(output.exact_norm_squared) &&
           std::isfinite(output.measure) && std::isfinite(output.direct_l2) &&
           std::isfinite(output.rms) && std::isfinite(output.relative_l2);
  };
  return publish(volume_error, volume_norm, volume_measure, 0, result.volume) &&
         publish(trace_error, trace_norm, trace_measure,
                 context.mesh->GetNumFaces(), result.trace);
}

bool VectorWithinScale(const std::vector<double>& first,
                       const std::vector<double>& second,
                       const double multiplier)
{
  const double scale = std::max({1.0, InfinityNorm(first), InfinityNorm(second)});
  return DifferenceNorm(first, second) <= multiplier * experiment_epsilon * scale;
}

double IndependentResidualNorm(const std::vector<double>& residual_interior,
                               const std::vector<double>& residual_trace,
                               const std::vector<int>& signed_trace_dofs,
                               const int global_trace_dofs)
{
  if (residual_trace.size() != signed_trace_dofs.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  long double squared = 0.0L;
  for (const double value : residual_interior)
  {
    squared += static_cast<long double>(value) * value;
  }
  std::vector<double> unique(static_cast<std::size_t>(global_trace_dofs), 0.0);
  for (std::size_t local = 0; local < residual_trace.size(); ++local)
  {
    double sign = 1.0;
    const int global = OracleDecodeSignedVDof(signed_trace_dofs[local], sign);
    if (global < 0 || global >= global_trace_dofs)
    {
      return std::numeric_limits<double>::infinity();
    }
    unique[global] += sign * residual_trace[local];
  }
  for (const double value : unique)
  {
    squared += static_cast<long double>(value) * value;
  }
  return std::sqrt(static_cast<double>(squared));
}

bool CheckProjectionAndInitialAdmissibility(
    const IndependentContext& context, const SolvedMeshEvidence& solved)
{
  std::vector<double> expected_old_interior, expected_old_trace,
      expected_initial_interior, expected_initial_trace;
  if (!IndependentProjection(context, false, expected_old_interior,
                             expected_old_trace) ||
      !IndependentProjection(context, true, expected_initial_interior,
                             expected_initial_trace))
  {
    return false;
  }
  if (!FiniteVector(expected_old_interior) || !FiniteVector(expected_old_trace) ||
      !FiniteVector(expected_initial_interior) ||
      !FiniteVector(expected_initial_trace) || !FiniteVector(solved.old_interior) ||
      !FiniteVector(solved.old_trace) || !FiniteVector(solved.initial_interior) ||
      !FiniteVector(solved.initial_trace) ||
      !VectorWithinScale(expected_old_interior, solved.old_interior, 4096.0) ||
      !VectorWithinScale(expected_old_trace, solved.old_trace, 4096.0) ||
      !VectorWithinScale(expected_initial_interior, solved.initial_interior,
                         4096.0) ||
      !VectorWithinScale(expected_initial_trace, solved.initial_trace, 4096.0))
  {
    return false;
  }
  const double interior_change = RelativeCoefficientTwoNorm(
      solved.old_interior, solved.initial_interior);
  const double trace_change = RelativeCoefficientTwoNorm(
      solved.old_trace, solved.initial_trace);
  if (!std::isfinite(interior_change) || !std::isfinite(trace_change) ||
      !(interior_change >= 1.0e-3) || !(trace_change >= 1.0e-3)) return false;
  OracleAdmissibility admissibility{};
  return IndependentPrescreen(context, solved.q_a, solved.initial_interior,
                              solved.initial_trace, admissibility) &&
         admissibility.passed && solved.newton.initial_prescreen_passed &&
         solved.initial_admissibility.passed &&
         solved.initial_admissibility.reason == TrialRejectionReason::None &&
         Close(admissibility.minimum_density,
               solved.initial_admissibility.minimum_density,
               4096.0 * experiment_epsilon) &&
         Close(admissibility.minimum_pressure,
               solved.initial_admissibility.minimum_pressure,
               4096.0 * experiment_epsilon) &&
         Close(admissibility.maximum_wave_speed,
               solved.initial_admissibility.maximum_wave_speed,
               4096.0 * experiment_epsilon) &&
         solved.newton.minimum_density <= admissibility.minimum_density +
             4096.0 * experiment_epsilon &&
         solved.newton.minimum_pressure <= admissibility.minimum_pressure +
             4096.0 * experiment_epsilon &&
         admissibility.maximum_wave_speed <= solved.newton.maximum_wave_speed +
             4096.0 * experiment_epsilon *
                 std::max(1.0, solved.newton.maximum_wave_speed) &&
         solved.newton.initial_residual > 1.0e-8;
}

bool CheckLocalAssemblyAndAllColumns(const IndependentContext& context,
                                     const SolvedMeshEvidence& solved)
{
  const LocalAssemblyAuditSnapshot& audit = solved.local_assembly;
  if (audit.element != 0 || audit.polynomial_order != solved.polynomial_order ||
      audit.elements_per_direction != solved.elements_per_direction ||
      audit.quadrature_order != solved.q_a || audit.interior_dofs <= 0 ||
      audit.trace_dofs <= 0 || audit.interior_reconstruction_id == 0 ||
      audit.trace_reconstruction_id == 0 ||
      audit.a.size() != static_cast<std::size_t>(audit.interior_dofs) *
                            audit.interior_dofs ||
      audit.b.size() != static_cast<std::size_t>(audit.interior_dofs) *
                            audit.trace_dofs ||
      audit.c.size() != static_cast<std::size_t>(audit.trace_dofs) *
                            audit.interior_dofs ||
      audit.d.size() != static_cast<std::size_t>(audit.trace_dofs) *
                            audit.trace_dofs ||
      !FiniteVector(audit.old_interior) ||
      !FiniteVector(audit.current_interior) || !FiniteVector(audit.trace) ||
      !FiniteVector(audit.residual_interior) ||
      !FiniteVector(audit.residual_trace) || !FiniteVector(audit.a) ||
      !FiniteVector(audit.b) || !FiniteVector(audit.c) ||
      !FiniteVector(audit.d) ||
      !(DifferenceNorm(audit.old_interior, audit.current_interior) > 0.0))
  {
    return false;
  }
  std::vector<double> base_r, base_g;
  if (!AssembleIndependentQOnSpaces(
          *context.interior_space, *context.trace_space, solved.q_a, 0,
          audit.current_interior, audit.old_interior, audit.trace, base_r,
          base_g) || !FiniteVector(base_r) || !FiniteVector(base_g))
  {
    return false;
  }
  std::vector<double> stored_q = audit.residual_interior;
  stored_q.insert(stored_q.end(), audit.residual_trace.begin(),
                  audit.residual_trace.end());
  std::vector<double> independent_q = base_r;
  independent_q.insert(independent_q.end(), base_g.begin(), base_g.end());
  const double s_q = std::max({1.0, InfinityNorm(stored_q),
                               InfinityNorm(independent_q)});
  if (!std::isfinite(s_q) ||
      DifferenceNorm(stored_q, independent_q) >
          2048.0 * experiment_epsilon * s_q)
  {
    return false;
  }
  const mfem::FiniteElement* fe = context.interior_space->GetFE(0);
  mfem::ElementTransformation* transformation =
      context.mesh->GetElementTransformation(0);
  if (fe == nullptr || transformation == nullptr) return false;
  std::vector<double> mass_residual(audit.interior_dofs, 0.0);
  mfem::Vector shape(fe->GetDof());
  const mfem::IntegrationRule& rule =
      mfem::IntRules.Get(fe->GetGeomType(), solved.q_a);
  for (int q = 0; q < rule.GetNPoints(); ++q)
  {
    const mfem::IntegrationPoint& point = rule.IntPoint(q);
    transformation->SetIntPoint(&point);
    fe->CalcShape(point, shape);
    const double weight = point.weight * transformation->Weight();
    ConservedState current_state{}, old_state{};
    if (!InterpolateOracleState(audit.current_interior, fe->GetDof(), 0, shape,
                                current_state) ||
        !InterpolateOracleState(audit.old_interior, fe->GetDof(), 0, shape,
                                old_state))
    {
      return false;
    }
    for (int row = 0; row < shape.Size(); ++row)
      for (int component = 0; component < conserved_components; ++component)
      {
        mass_residual[row + shape.Size() * component] += weight * shape[row] *
            (current_state[component] - old_state[component]) /
            backward_euler_dt;
      }
  }
  if (!FiniteVector(mass_residual) || !(InfinityNorm(mass_residual) > 0.0))
  {
    return false;
  }

  const double h = std::cbrt(experiment_epsilon) * std::max(
      {1.0, InfinityNorm(audit.current_interior), InfinityNorm(audit.trace)});
  if (!(h > 0.0)) return false;
  mfem::Array<int> interior_vdofs;
  context.interior_space->GetElementVDofs(0, interior_vdofs);
  if (interior_vdofs.Size() != audit.interior_dofs ||
      audit.signed_trace_dofs.size() != static_cast<std::size_t>(audit.trace_dofs))
  {
    return false;
  }
  for (int column = 0; column < audit.interior_dofs; ++column)
  {
    std::vector<double> plus = audit.current_interior;
    std::vector<double> minus = audit.current_interior;
    plus[column] += h;
    minus[column] -= h;
    double global_sign = 1.0;
    const int global_column = OracleDecodeSignedVDof(
        interior_vdofs[column], global_sign);
    std::vector<double> global_plus = solved.initial_interior;
    std::vector<double> global_minus = solved.initial_interior;
    global_plus[global_column] += global_sign * h;
    global_minus[global_column] -= global_sign * h;
    OracleAdmissibility plus_admissibility{}, minus_admissibility{};
    if (!IndependentPrescreen(context, solved.q_a, global_plus,
                              solved.initial_trace, plus_admissibility) ||
        !IndependentPrescreen(context, solved.q_a, global_minus,
                              solved.initial_trace, minus_admissibility) ||
        !plus_admissibility.passed || !minus_admissibility.passed)
    {
      return false;
    }
    std::vector<double> plus_r, plus_g, minus_r, minus_g;
    if (!AssembleIndependentQOnSpaces(
            *context.interior_space, *context.trace_space, solved.q_a, 0,
            plus, audit.old_interior, audit.trace, plus_r, plus_g) ||
        !AssembleIndependentQOnSpaces(
            *context.interior_space, *context.trace_space, solved.q_a, 0,
            minus, audit.old_interior, audit.trace, minus_r, minus_g))
    {
      return false;
    }
    std::vector<double> finite_difference, jacobian_column;
    finite_difference.reserve(audit.interior_dofs + audit.trace_dofs);
    jacobian_column.reserve(audit.interior_dofs + audit.trace_dofs);
    for (int row = 0; row < audit.interior_dofs; ++row)
    {
      finite_difference.push_back((plus_r[row] - minus_r[row]) / (2.0 * h));
      jacobian_column.push_back(audit.a[
          static_cast<std::size_t>(row) * audit.interior_dofs + column]);
    }
    for (int row = 0; row < audit.trace_dofs; ++row)
    {
      finite_difference.push_back((plus_g[row] - minus_g[row]) / (2.0 * h));
      jacobian_column.push_back(audit.c[
          static_cast<std::size_t>(row) * audit.interior_dofs + column]);
    }
    const double s_j = std::max({1.0, InfinityNorm(jacobian_column),
                                 InfinityNorm(finite_difference)});
    if (!FiniteVector(plus_r) || !FiniteVector(plus_g) ||
        !FiniteVector(minus_r) || !FiniteVector(minus_g) ||
        !std::isfinite(s_j) ||
        DifferenceNorm(jacobian_column, finite_difference) >
            2.0e-6 * s_j + 256.0 * experiment_epsilon * s_q / h)
    {
      return false;
    }
  }
  for (int column = 0; column < audit.trace_dofs; ++column)
  {
    std::vector<double> plus = audit.trace;
    std::vector<double> minus = audit.trace;
    plus[column] += h;
    minus[column] -= h;
    double global_sign = 1.0;
    const int global_column = OracleDecodeSignedVDof(
        audit.signed_trace_dofs[column], global_sign);
    std::vector<double> global_plus = solved.initial_trace;
    std::vector<double> global_minus = solved.initial_trace;
    global_plus[global_column] += global_sign * h;
    global_minus[global_column] -= global_sign * h;
    OracleAdmissibility plus_admissibility{}, minus_admissibility{};
    if (!IndependentPrescreen(context, solved.q_a, solved.initial_interior,
                              global_plus, plus_admissibility) ||
        !IndependentPrescreen(context, solved.q_a, solved.initial_interior,
                              global_minus, minus_admissibility) ||
        !plus_admissibility.passed || !minus_admissibility.passed)
    {
      return false;
    }
    std::vector<double> plus_r, plus_g, minus_r, minus_g;
    if (!AssembleIndependentQOnSpaces(
            *context.interior_space, *context.trace_space, solved.q_a, 0,
            audit.current_interior, audit.old_interior, plus, plus_r, plus_g) ||
        !AssembleIndependentQOnSpaces(
            *context.interior_space, *context.trace_space, solved.q_a, 0,
            audit.current_interior, audit.old_interior, minus, minus_r, minus_g))
    {
      return false;
    }
    std::vector<double> finite_difference, jacobian_column;
    finite_difference.reserve(audit.interior_dofs + audit.trace_dofs);
    jacobian_column.reserve(audit.interior_dofs + audit.trace_dofs);
    for (int row = 0; row < audit.interior_dofs; ++row)
    {
      finite_difference.push_back((plus_r[row] - minus_r[row]) / (2.0 * h));
      jacobian_column.push_back(audit.b[
          static_cast<std::size_t>(row) * audit.trace_dofs + column]);
    }
    for (int row = 0; row < audit.trace_dofs; ++row)
    {
      finite_difference.push_back((plus_g[row] - minus_g[row]) / (2.0 * h));
      jacobian_column.push_back(audit.d[
          static_cast<std::size_t>(row) * audit.trace_dofs + column]);
    }
    const double s_j = std::max({1.0, InfinityNorm(jacobian_column),
                                 InfinityNorm(finite_difference)});
    if (!FiniteVector(plus_r) || !FiniteVector(plus_g) ||
        !FiniteVector(minus_r) || !FiniteVector(minus_g) ||
        !std::isfinite(s_j) ||
        DifferenceNorm(jacobian_column, finite_difference) >
            2.0e-6 * s_j + 256.0 * experiment_epsilon * s_q / h)
    {
      return false;
    }
  }
  return true;
}

bool CheckGlobalResidualsAndTrials(const IndependentContext& context,
                                   const SolvedMeshEvidence& solved)
{
  if (!FiniteVector(solved.old_interior) || !FiniteVector(solved.old_trace) ||
      !FiniteVector(solved.initial_interior) ||
      !FiniteVector(solved.initial_trace) || !FiniteVector(solved.final_interior) ||
      !FiniteVector(solved.final_trace) ||
      !FiniteVector(solved.initial_residual_interior) ||
      !FiniteVector(solved.initial_residual_trace) ||
      !FiniteVector(solved.final_residual_interior) ||
      !FiniteVector(solved.final_residual_trace))
  {
    return false;
  }
  std::vector<double> initial_r, initial_g, final_r, final_g;
  std::vector<int> initial_map, final_map;
  if (!AssembleIndependentGlobalQ(
          context, solved.q_a, solved.old_interior, solved.initial_interior,
          solved.initial_trace, initial_r, initial_g, initial_map) ||
      !AssembleIndependentGlobalQ(
          context, solved.q_a, solved.old_interior, solved.final_interior,
          solved.final_trace, final_r, final_g, final_map) ||
      initial_map != solved.signed_trace_dofs || final_map != initial_map ||
      !VectorWithinScale(initial_r, solved.initial_residual_interior, 4096.0) ||
      !VectorWithinScale(initial_g, solved.initial_residual_trace, 4096.0) ||
      !VectorWithinScale(final_r, solved.final_residual_interior, 4096.0) ||
      !VectorWithinScale(final_g, solved.final_residual_trace, 4096.0))
  {
    return false;
  }
  const double initial_norm = IndependentResidualNorm(
      initial_r, initial_g, initial_map, context.trace_space->GetVSize());
  const double final_norm = IndependentResidualNorm(
      final_r, final_g, final_map, context.trace_space->GetVSize());
  const double coefficient_change = ConcatenatedCoefficientRelativeTwoNorm(
      solved.initial_interior, solved.final_interior, solved.initial_trace,
      solved.final_trace);
  const TrialAuditSnapshot* first_accepted = nullptr;
  for (const TrialAuditSnapshot& audit : solved.trial_audits)
    if (audit.accepted)
    {
      if (first_accepted != nullptr) return false;
      first_accepted = &audit;
    }
  if (first_accepted == nullptr) return false;
  std::vector<double> initial_jacobian, accepted_jacobian;
  for (int element = 0; element < context.mesh->GetNE(); ++element)
  {
    std::vector<double> initial_local, initial_trace_local, accepted_local,
        accepted_trace_local, initial_a, initial_b, initial_c, initial_d,
        accepted_a, accepted_b, accepted_c, accepted_d;
    if (!ExtractIndependentElement(context, solved.initial_interior,
                                   solved.initial_trace, element, initial_local,
                                   initial_trace_local) ||
        !ExtractIndependentElement(context, first_accepted->trial_interior,
                                   first_accepted->trial_trace, element,
                                   accepted_local, accepted_trace_local) ||
        !AssembleIndependentAnalyticBlocks(
            *context.interior_space, *context.trace_space, solved.q_a, element,
            initial_local, initial_trace_local, initial_a, initial_b, initial_c,
            initial_d) ||
        !AssembleIndependentAnalyticBlocks(
            *context.interior_space, *context.trace_space, solved.q_a, element,
            accepted_local, accepted_trace_local, accepted_a, accepted_b,
            accepted_c, accepted_d))
    {
      return false;
    }
    const auto append = [](std::vector<double>& target,
                           const std::vector<double>& source)
    {
      target.insert(target.end(), source.begin(), source.end());
    };
    append(initial_jacobian, initial_a);
    append(initial_jacobian, initial_b);
    append(initial_jacobian, initial_c);
    append(initial_jacobian, initial_d);
    append(accepted_jacobian, accepted_a);
    append(accepted_jacobian, accepted_b);
    append(accepted_jacobian, accepted_c);
    append(accepted_jacobian, accepted_d);
  }
  const double jacobian_change = RelativeCoefficientTwoNorm(
      initial_jacobian, accepted_jacobian);
  if (solved.newton.status != ExperimentStatus::Success ||
      !solved.newton.admissible ||
      !std::isfinite(solved.newton.initial_residual) ||
      !std::isfinite(solved.newton.final_residual) ||
      !std::isfinite(solved.newton.trace_relative_residual) ||
      !std::isfinite(solved.newton.minimum_density) ||
      !std::isfinite(solved.newton.minimum_pressure) ||
      !std::isfinite(solved.newton.minimum_damping) ||
      !std::isfinite(solved.newton.maximum_wave_speed) ||
      !std::isfinite(solved.newton.maximum_coefficient_update) ||
      !std::isfinite(solved.newton.maximum_jacobian_change) ||
      !std::isfinite(solved.newton.relative_coefficient_change) ||
      !std::isfinite(solved.newton.relative_jacobian_change) ||
      !std::isfinite(solved.newton.residual_after_first_accepted) ||
      !Close(initial_norm, solved.newton.initial_residual,
             4096.0 * experiment_epsilon) ||
      !Close(final_norm, solved.newton.final_residual,
             4096.0 * experiment_epsilon) ||
      solved.newton.nonlinear_iterations < 2 ||
      solved.newton.linear_iterations < 1 ||
      solved.residual_history.size() !=
          static_cast<std::size_t>(solved.newton.nonlinear_iterations + 1) ||
      solved.linear_history.size() !=
          static_cast<std::size_t>(solved.newton.nonlinear_iterations) ||
      !std::isfinite(coefficient_change) || coefficient_change < 1.0e-6 ||
      !std::isfinite(jacobian_change) || jacobian_change < 1.0e-8 ||
      !Close(coefficient_change, solved.newton.relative_coefficient_change,
             4096.0 * experiment_epsilon) ||
      !Close(jacobian_change, solved.newton.relative_jacobian_change,
             4096.0 * experiment_epsilon) ||
      !(final_norm <= 1.0e-10 * (1.0 + initial_norm)))
  {
    return false;
  }
  for (const auto& linear : solved.linear_history)
  {
    if (linear.status != ExperimentStatus::Success ||
        linear.gmres_iterations < 0 ||
        !std::isfinite(linear.right_hand_side_norm) ||
        !(linear.right_hand_side_norm >= 0.0) ||
        !std::isfinite(linear.reported_relative_residual) ||
        linear.reported_relative_residual < 0.0 ||
        !std::isfinite(linear.true_relative_residual) ||
        !(linear.true_relative_residual >= 0.0) ||
        !(linear.true_relative_residual <= 1.0e-10)) return false;
  }

  if (!FiniteVector(solved.residual_history)) return false;
  int accepted_history = 0, rejected_history = 0, reassembly_history = 0;
  int linear_iterations = 0;
  for (const auto& linear : solved.linear_history)
  {
    linear_iterations += linear.gmres_iterations;
  }
  double minimum_damping = 1.0;
  double minimum_density = solved.initial_admissibility.minimum_density;
  double minimum_pressure = solved.initial_admissibility.minimum_pressure;
  double maximum_wave_speed = solved.initial_admissibility.maximum_wave_speed;
  double maximum_update = 0.0, maximum_jacobian = 0.0;
  int expected_trial_index = 0;
  for (const auto& trial : solved.trial_history)
  {
    if (trial.status != ExperimentStatus::Success ||
        !std::isfinite(trial.alpha) || !std::isfinite(trial.base_residual) ||
        !std::isfinite(trial.armijo_boundary) ||
        !std::isfinite(trial.coefficient_update) ||
        !std::isfinite(trial.jacobian_change) ||
        !std::isfinite(trial.admissibility.minimum_density) ||
        !std::isfinite(trial.admissibility.minimum_pressure) ||
        !std::isfinite(trial.admissibility.maximum_wave_speed) ||
        trial.nonlinear_iteration != accepted_history ||
        trial.trial_index != expected_trial_index ||
        trial.trial_index < 0 || trial.trial_index > 12 ||
        trial.alpha != std::ldexp(1.0, -trial.trial_index) ||
        trial.armijo_boundary !=
            (1.0 - 1.0e-4 * trial.alpha) * trial.base_residual ||
        !Close(trial.base_residual, solved.residual_history[accepted_history],
               4096.0 * experiment_epsilon)) return false;
    if (trial.reassembly_attempted) ++reassembly_history;
    if (trial.accepted)
    {
      ++accepted_history;
      if (trial.rejection_reason != TrialRejectionReason::None ||
          !trial.admissibility.passed || !trial.reassembly_attempted ||
          !trial.residual_evaluated || !trial.trial_residual_present ||
          !trial.generation_published || trial.generation_discarded ||
          !std::isfinite(trial.trial_residual) ||
          trial.trial_residual > trial.armijo_boundary) return false;
      if (!Close(trial.base_residual,
                 solved.residual_history[accepted_history - 1],
                 4096.0 * experiment_epsilon) ||
          !Close(trial.trial_residual,
                 solved.residual_history[accepted_history],
                 4096.0 * experiment_epsilon)) return false;
      minimum_damping = std::min(minimum_damping, trial.alpha);
      minimum_density = std::min(minimum_density,
                                 trial.admissibility.minimum_density);
      minimum_pressure = std::min(minimum_pressure,
                                  trial.admissibility.minimum_pressure);
      maximum_wave_speed = std::max(maximum_wave_speed,
                                    trial.admissibility.maximum_wave_speed);
      maximum_update = std::max(maximum_update, trial.coefficient_update);
      maximum_jacobian = std::max(maximum_jacobian, trial.jacobian_change);
      expected_trial_index = 0;
    }
    else if (trial.rejection_reason == TrialRejectionReason::Armijo)
    {
      ++rejected_history;
      if (!trial.admissibility.passed || !trial.reassembly_attempted ||
          !trial.residual_evaluated || !trial.trial_residual_present ||
          !std::isfinite(trial.trial_residual) ||
          trial.trial_residual <= trial.armijo_boundary ||
          !trial.generation_discarded || trial.generation_published) return false;
      ++expected_trial_index;
    }
    else
    {
      ++rejected_history;
      if (trial.admissibility.passed || trial.reassembly_attempted ||
          trial.residual_evaluated || trial.trial_residual_present ||
          !trial.generation_discarded || trial.generation_published) return false;
      ++expected_trial_index;
    }
  }
  if (accepted_history != solved.newton.nonlinear_iterations ||
      rejected_history != solved.newton.rejected_trials ||
      reassembly_history + 1 != solved.newton.assembly_attempts ||
      linear_iterations != solved.newton.linear_iterations ||
      !Close(minimum_damping, solved.newton.minimum_damping,
             256.0 * experiment_epsilon) ||
      !Close(minimum_density, solved.newton.minimum_density,
             4096.0 * experiment_epsilon) ||
      !Close(minimum_pressure, solved.newton.minimum_pressure,
             4096.0 * experiment_epsilon) ||
      !Close(maximum_wave_speed, solved.newton.maximum_wave_speed,
             4096.0 * experiment_epsilon) ||
      !Close(maximum_update, solved.newton.maximum_coefficient_update,
             256.0 * experiment_epsilon) ||
      !Close(maximum_jacobian, solved.newton.maximum_jacobian_change,
             256.0 * experiment_epsilon))
  {
    return false;
  }

  int accepted_audits = 0, rejected_audits = 0;
  for (const TrialAuditSnapshot& audit : solved.trial_audits)
  {
    if (audit.status != ExperimentStatus::Success ||
        audit.base_interior.size() != audit.delta_interior.size() ||
        audit.base_trace.size() != audit.delta_trace.size() ||
        audit.trial_interior.size() != audit.base_interior.size() ||
        audit.trial_trace.size() != audit.base_trace.size() ||
        audit.post_decision_interior.size() != audit.base_interior.size() ||
        audit.post_decision_trace.size() != audit.base_trace.size() ||
        !FiniteVector(audit.base_interior) || !FiniteVector(audit.base_trace) ||
        !FiniteVector(audit.delta_interior) || !FiniteVector(audit.delta_trace) ||
        !FiniteVector(audit.trial_interior) || !FiniteVector(audit.trial_trace) ||
        !FiniteVector(audit.post_decision_interior) ||
        !FiniteVector(audit.post_decision_trace) || !std::isfinite(audit.alpha) ||
        !std::isfinite(audit.base_residual) ||
        !std::isfinite(audit.armijo_boundary) ||
        !std::isfinite(audit.coefficient_update) ||
        !std::isfinite(audit.jacobian_change) ||
        !std::isfinite(audit.formula_identity_error)) return false;
    const double scale = std::max({1.0, InfinityNorm(audit.base_interior),
                                   InfinityNorm(audit.base_trace),
                                   InfinityNorm(audit.trial_interior),
                                   InfinityNorm(audit.trial_trace)});
    if (audit.trial_index < 0 || audit.trial_index > 12 ||
        audit.alpha != std::ldexp(1.0, -audit.trial_index) ||
        audit.armijo_boundary !=
            (1.0 - 1.0e-4 * audit.alpha) * audit.base_residual) return false;
    double identity_error = 0.0;
    for (std::size_t index = 0; index < audit.base_interior.size(); ++index)
    {
      identity_error = std::max(identity_error, std::abs(
          audit.trial_interior[index] -
          (audit.base_interior[index] + audit.alpha * audit.delta_interior[index])));
    }
    for (std::size_t index = 0; index < audit.base_trace.size(); ++index)
    {
      identity_error = std::max(identity_error, std::abs(
          audit.trial_trace[index] -
          (audit.base_trace[index] + audit.alpha * audit.delta_trace[index])));
    }
    if (identity_error > 64.0 * experiment_epsilon * scale ||
        audit.formula_identity_error > 64.0 * experiment_epsilon * scale)
    {
      return false;
    }
    const double independent_update = std::max(
        DifferenceNorm(audit.base_interior, audit.trial_interior),
        DifferenceNorm(audit.base_trace, audit.trial_trace));
    if (!Close(independent_update, audit.coefficient_update,
               256.0 * experiment_epsilon)) return false;
    OracleAdmissibility admissibility{};
    if (!IndependentPrescreen(context, solved.q_a, audit.trial_interior,
                              audit.trial_trace, admissibility) ||
        admissibility.passed != audit.admissibility.passed ||
        admissibility.reason != audit.admissibility.reason ||
        !Close(admissibility.minimum_density,
               audit.admissibility.minimum_density,
               4096.0 * experiment_epsilon) ||
        !Close(admissibility.minimum_pressure,
               audit.admissibility.minimum_pressure,
               4096.0 * experiment_epsilon) ||
        !Close(admissibility.maximum_wave_speed,
               audit.admissibility.maximum_wave_speed,
               4096.0 * experiment_epsilon)) return false;
    if (audit.residual_evaluated)
    {
      std::vector<double> trial_r, trial_g;
      std::vector<int> trial_map;
      if (!AssembleIndependentGlobalQ(
              context, solved.q_a, solved.old_interior, audit.trial_interior,
              audit.trial_trace, trial_r, trial_g, trial_map)) return false;
      const double trial_norm = IndependentResidualNorm(
          trial_r, trial_g, trial_map, context.trace_space->GetVSize());
      if (!Close(trial_norm, audit.trial_residual,
                 4096.0 * experiment_epsilon)) return false;
    }
    if (audit.accepted)
    {
      ++accepted_audits;
      if (audit.rejection_reason != TrialRejectionReason::None ||
          !audit.admissibility.passed || !audit.reassembly_attempted ||
          !audit.residual_evaluated || !std::isfinite(audit.trial_residual) ||
          !audit.generation_published || audit.generation_discarded ||
          audit.post_decision_interior != audit.trial_interior ||
          audit.post_decision_trace != audit.trial_trace ||
          audit.trial_residual > audit.armijo_boundary) return false;
    }
    else
    {
      ++rejected_audits;
      if (!audit.generation_discarded || audit.generation_published ||
          audit.post_decision_interior != audit.base_interior ||
          audit.post_decision_trace != audit.base_trace) return false;
      if (audit.rejection_reason == TrialRejectionReason::Armijo)
      {
        if (!audit.admissibility.passed || !audit.reassembly_attempted ||
            !audit.residual_evaluated || !std::isfinite(audit.trial_residual) ||
            !(audit.trial_residual > audit.armijo_boundary)) return false;
      }
      else if (audit.rejection_reason == TrialRejectionReason::None ||
               audit.admissibility.passed || audit.reassembly_attempted ||
               audit.residual_evaluated ||
               audit.rejection_reason != audit.admissibility.reason)
      {
        return false;
      }
    }
  }
  return accepted_audits == 1 &&
         rejected_audits == (solved.newton.rejected_trials > 0 ? 1 : 0) &&
         solved.trial_audits.size() ==
             static_cast<std::size_t>(1 + rejected_audits);
}

bool CheckLinearAudit(const SolvedMeshEvidence& solved)
{
  if (!solved.has_linear_audit) return solved.elements_per_direction != 3;
  const LinearAuditSnapshot& audit = solved.linear_audit;
  if (solved.elements_per_direction != 3 || audit.rows <= 0 ||
      audit.row_offsets.size() != static_cast<std::size_t>(audit.rows + 1) ||
      audit.columns.size() != audit.values.size() ||
      audit.right_hand_side.size() != static_cast<std::size_t>(audit.rows) ||
      audit.trace_increment.size() != static_cast<std::size_t>(audit.rows) ||
      audit.row_offsets.front() != 0 ||
      audit.row_offsets.back() != static_cast<int>(audit.values.size()))
  {
    return false;
  }
  std::vector<double> residual = audit.right_hand_side;
  for (double& value : residual) value = -value;
  for (int row = 0; row < audit.rows; ++row)
    for (int entry = audit.row_offsets[row]; entry < audit.row_offsets[row + 1];
         ++entry)
    {
      const int column = audit.columns[entry];
      if (column < 0 || column >= audit.rows) return false;
      residual[row] += audit.values[entry] * audit.trace_increment[column];
    }
  long double residual_squared = 0.0L, rhs_squared = 0.0L;
  for (int row = 0; row < audit.rows; ++row)
  {
    residual_squared += static_cast<long double>(residual[row]) * residual[row];
    rhs_squared += static_cast<long double>(audit.right_hand_side[row]) *
                   audit.right_hand_side[row];
  }
  const double relative = std::sqrt(static_cast<double>(residual_squared)) /
      std::max(std::sqrt(static_cast<double>(rhs_squared)), 1.0e-300);
  return relative <= 1.0e-10 &&
         Close(relative, audit.true_relative_residual,
               4096.0 * experiment_epsilon);
}

struct IndependentCampaignOracles
{
  std::array<std::array<ContinuousOracleResult, 3>, 4> solves{};
  std::array<std::array<double, 2>, 4> volume_rates{};
  std::array<std::array<double, 2>, 4> trace_rates{};
};

bool ErrorIntegralIsValid(const ErrorIntegralEvidence& value,
                          const double expected_measure,
                          const int expected_faces)
{
  if (!std::isfinite(value.error_squared) ||
      !std::isfinite(value.exact_norm_squared) ||
      !std::isfinite(value.measure) || !std::isfinite(value.direct_l2) ||
      !std::isfinite(value.rms) || !std::isfinite(value.relative_l2) ||
      !(value.error_squared > 0.0) || !(value.exact_norm_squared > 0.0) ||
      !(value.measure > 0.0) || !(value.direct_l2 > 0.0) ||
      !(value.rms > 0.0) || !(value.relative_l2 >= 1.0e-10) ||
      value.unique_face_count != expected_faces ||
      std::abs(value.measure - expected_measure) >
          128.0 * experiment_epsilon * std::max(1.0, expected_measure))
  {
    return false;
  }
  const double scale = std::max(1.0, value.error_squared);
  return std::abs(value.direct_l2 * value.direct_l2 - value.error_squared) <=
             128.0 * experiment_epsilon * scale &&
         std::abs(value.rms * value.rms * value.measure -
                  value.error_squared) <=
             128.0 * experiment_epsilon * scale &&
         std::abs(value.relative_l2 * value.relative_l2 *
                      value.exact_norm_squared - value.error_squared) <=
             128.0 * experiment_epsilon * scale;
}

bool CheckContinuousErrors(const ExperimentEvidence& evidence,
                           IndependentCampaignOracles& oracles)
{
  if (evidence.convergence.size() != 4) return false;
  for (std::size_t p_index = 0; p_index < evidence.convergence.size(); ++p_index)
  {
    const ConvergenceEvidence& ladder = evidence.convergence[p_index];
    if (ladder.polynomial_order != static_cast<int>(p_index) + 1 ||
        ladder.status != ExperimentStatus::Success || !ladder.passed) return false;
    for (std::size_t mesh_index = 0; mesh_index < 3; ++mesh_index)
    {
      const SolvedMeshEvidence& solved = ladder.solves[mesh_index];
      const double expected_trace_measure =
          2.0 * solved.elements_per_direction;
      const int expected_faces = 2 * solved.elements_per_direction *
                                 solved.elements_per_direction;
      const auto quadrature_pair_passes = [](
          const ErrorIntegralEvidence& q_e,
          const ErrorIntegralEvidence& q_e_plus_four)
      {
        const double difference = std::abs(q_e.relative_l2 -
                                           q_e_plus_four.relative_l2);
        return q_e.relative_l2 >= 1.0e-10 &&
               q_e_plus_four.relative_l2 >= 1.0e-10 &&
               difference / std::max(q_e_plus_four.relative_l2, 1.0e-300) <=
                   1.0e-8 &&
               q_e_plus_four.relative_l2 > 100.0 * difference;
      };
      if (!ErrorIntegralIsValid(solved.q_e_volume, 1.0, 0) ||
          !ErrorIntegralIsValid(solved.q_e_plus_four_volume, 1.0, 0) ||
          !ErrorIntegralIsValid(solved.q_e_trace, expected_trace_measure,
                                expected_faces) ||
          !ErrorIntegralIsValid(solved.q_e_plus_four_trace,
                                expected_trace_measure, expected_faces) ||
          !quadrature_pair_passes(solved.q_e_volume,
                                  solved.q_e_plus_four_volume) ||
          !quadrature_pair_passes(solved.q_e_trace,
                                  solved.q_e_plus_four_trace) ||
          !std::isfinite(ladder.volume_relative_l2[mesh_index]) ||
          !std::isfinite(ladder.trace_relative_l2[mesh_index]) ||
          ladder.volume_relative_l2[mesh_index] !=
              solved.q_e_plus_four_volume.relative_l2 ||
          ladder.trace_relative_l2[mesh_index] !=
              solved.q_e_plus_four_trace.relative_l2)
      {
        return false;
      }
      IndependentContext context;
      if (!BuildIndependentContext(solved.polynomial_order,
                                   solved.elements_per_direction, context) ||
          !IndependentContinuousErrors(
              context, solved.final_interior, solved.final_trace,
              5 * solved.polynomial_order + 16,
              oracles.solves[p_index][mesh_index])) return false;
      const ContinuousOracleResult& q_o = oracles.solves[p_index][mesh_index];
      const auto raw_agrees = [](const ErrorIntegralEvidence& production,
                                 const ErrorIntegralEvidence& oracle)
      {
        const double error_scale = std::max(
            {std::abs(production.error_squared),
             std::abs(oracle.error_squared), 1.0e-300});
        const double norm_scale = std::max(
            {std::abs(production.exact_norm_squared),
             std::abs(oracle.exact_norm_squared), 1.0e-300});
        const double relative_scale = std::max(
            {std::abs(production.relative_l2),
             std::abs(oracle.relative_l2), 1.0e-300});
        return std::abs(production.error_squared - oracle.error_squared) /
                       error_scale <= 2.0e-8 &&
               std::abs(production.exact_norm_squared -
                        oracle.exact_norm_squared) / norm_scale <= 2.0e-8 &&
               std::abs(production.relative_l2 - oracle.relative_l2) /
                       relative_scale <= 2.0e-8;
      };
      if (!ErrorIntegralIsValid(q_o.volume, 1.0, 0) ||
          !ErrorIntegralIsValid(q_o.trace, expected_trace_measure,
                                expected_faces) ||
          !raw_agrees(solved.q_e_plus_four_volume, q_o.volume) ||
          !raw_agrees(solved.q_e_plus_four_trace, q_o.trace) ||
          std::abs(q_o.volume.measure -
                   solved.q_e_plus_four_volume.measure) >
              256.0 * experiment_epsilon *
                  std::max(1.0, expected_trace_measure) ||
          std::abs(q_o.trace.measure - solved.q_e_plus_four_trace.measure) >
              256.0 * experiment_epsilon *
                  std::max(1.0, expected_trace_measure) ||
          !std::isfinite(q_o.maximum_face_measure_error) ||
          q_o.maximum_face_measure_error >
              256.0 * experiment_epsilon *
                  std::max(1.0, 1.0 / solved.elements_per_direction) ||
          q_o.trace.unique_face_count != expected_faces)
      {
        return false;
      }
    }
    for (int interval = 0; interval < 2; ++interval)
    {
      oracles.volume_rates[p_index][interval] = std::log2(
          oracles.solves[p_index][interval].volume.relative_l2 /
          oracles.solves[p_index][interval + 1].volume.relative_l2);
      oracles.trace_rates[p_index][interval] = std::log2(
          oracles.solves[p_index][interval].trace.relative_l2 /
          oracles.solves[p_index][interval + 1].trace.relative_l2);
      const double recorded_volume_rate = std::log2(
          ladder.volume_relative_l2[interval] /
          ladder.volume_relative_l2[interval + 1]);
      const double recorded_trace_rate = std::log2(
          ladder.trace_relative_l2[interval] /
          ladder.trace_relative_l2[interval + 1]);
      if (!std::isfinite(oracles.volume_rates[p_index][interval]) ||
          !std::isfinite(oracles.trace_rates[p_index][interval]) ||
          !std::isfinite(ladder.volume_rates[interval]) ||
          !std::isfinite(ladder.trace_rates[interval]) ||
          ladder.volume_rates[interval] != recorded_volume_rate ||
          ladder.trace_rates[interval] != recorded_trace_rate ||
          std::abs(oracles.volume_rates[p_index][interval] -
                   ladder.volume_rates[interval]) > 1.0e-7 ||
          std::abs(oracles.trace_rates[p_index][interval] -
                   ladder.trace_rates[interval]) > 1.0e-7) return false;
    }
  }
  return true;
}

void SetCoupledRelativeError(ErrorIntegralEvidence& error,
                             const double relative)
{
  error.relative_l2 = relative;
  error.error_squared = relative * relative * error.exact_norm_squared;
  error.direct_l2 = std::sqrt(error.error_squared);
  error.rms = std::sqrt(error.error_squared / error.measure);
}

std::string ValidateEvidenceCandidate(
    const ExperimentEvidence& evidence,
    const IndependentCampaignOracles& oracles)
{
  if (evidence.rollback.entry_interior != evidence.rollback.restored_interior ||
      evidence.rollback.entry_trace != evidence.rollback.restored_trace ||
      evidence.rollback.entry_interior_hash !=
          evidence.rollback.restored_interior_hash ||
      evidence.rollback.entry_trace_hash != evidence.rollback.restored_trace_hash)
  {
    return "RollbackVectorMismatch";
  }
  if (!std::isfinite(evidence.rollback.initial_residual) ||
      !std::isfinite(evidence.rollback.residual_after_first_accepted) ||
      !evidence.rollback.residual_after_first_above_convergence ||
      !(evidence.rollback.residual_after_first_accepted >
        1.0e-10 * (1.0 + evidence.rollback.initial_residual)))
  {
    return "RollbackResidualProofMismatch";
  }
  if (!evidence.rollback.local_generation_invalid ||
      !evidence.rollback.condensed_generation_invalid ||
      !evidence.rollback.solve_generation_invalid ||
      !evidence.rollback.local_copy_rejected ||
      !evidence.rollback.right_hand_side_copy_rejected ||
      !evidence.rollback.trace_increment_copy_rejected ||
      !evidence.rollback.condensation_consume_rejected ||
      !evidence.rollback.solve_consume_rejected ||
      !evidence.rollback.recovery_consume_rejected ||
      evidence.rollback.partial_solve_evidence ||
      evidence.rollback.partial_convergence_evidence)
  {
    return "RollbackConsumeFlagMismatch";
  }
  for (std::size_t p_index = 0; p_index < evidence.convergence.size(); ++p_index)
  {
    const ConvergenceEvidence& ladder = evidence.convergence[p_index];
    for (std::size_t mesh_index = 0; mesh_index < 3; ++mesh_index)
    {
      const SolvedMeshEvidence& solved = ladder.solves[mesh_index];
      if (solved.final_interior == solved.initial_interior &&
          solved.final_trace == solved.initial_trace)
      {
        return "NoUpdateFinalEqualsInitial";
      }
      const ErrorIntegralEvidence& skeleton = solved.q_e_plus_four_trace;
      const double relation_scale = std::max(1.0, skeleton.error_squared);
      if (std::abs(skeleton.direct_l2 * skeleton.direct_l2 -
                   skeleton.error_squared) >
          64.0 * experiment_epsilon * relation_scale)
      {
        return "SkeletonNumeratorMismatch";
      }
      if (std::abs(skeleton.relative_l2 * skeleton.relative_l2 *
                       skeleton.exact_norm_squared -
                   skeleton.error_squared) >
          64.0 * experiment_epsilon * relation_scale)
      {
        return "SkeletonDenominatorMismatch";
      }
      if (std::abs(skeleton.rms * skeleton.rms * skeleton.measure -
                   skeleton.error_squared) >
          64.0 * experiment_epsilon * relation_scale)
      {
        return "SkeletonMeasureMismatch";
      }
      int accepted = 0;
      for (const TrialAuditSnapshot& audit : solved.trial_audits)
      {
        if (audit.accepted &&
            (!std::isfinite(audit.trial_residual) ||
             !std::isfinite(audit.armijo_boundary) ||
             audit.trial_residual > audit.armijo_boundary))
        {
          return "ArmijoBoundaryViolation";
        }
      }
      for (const auto& trial : solved.trial_history)
      {
        if (trial.accepted)
        {
          ++accepted;
          if (trial.trial_residual > trial.armijo_boundary)
          {
            return "ArmijoBoundaryViolation";
          }
        }
      }
      if (accepted != solved.newton.nonlinear_iterations)
      {
        return "ArmijoRecordCountMismatch";
      }
    }
  }
  if (evidence.convergence.size() > 1)
  {
    const ConvergenceEvidence& p_two = evidence.convergence[1];
    bool assigned = true;
    for (std::size_t mesh_index = 0; mesh_index < 3; ++mesh_index)
    {
      const double target = 1.0e-3 * std::pow(
          3.0 / p_two.mesh_widths[mesh_index], 3.0);
      assigned = assigned && Close(
          p_two.solves[mesh_index].q_e_volume.relative_l2, target, 64.0 *
              experiment_epsilon) &&
          Close(p_two.solves[mesh_index].q_e_plus_four_volume.relative_l2,
                target, 64.0 * experiment_epsilon) &&
          Close(p_two.solves[mesh_index].q_e_trace.relative_l2, target,
                64.0 * experiment_epsilon) &&
          Close(p_two.solves[mesh_index].q_e_plus_four_trace.relative_l2,
                target, 64.0 * experiment_epsilon);
    }
    assigned = assigned && Close(p_two.volume_rates[0], 3.0,
                                 64.0 * experiment_epsilon) &&
        Close(p_two.volume_rates[1], 3.0, 64.0 * experiment_epsilon) &&
        Close(p_two.trace_rates[0], 3.0, 64.0 * experiment_epsilon) &&
        Close(p_two.trace_rates[1], 3.0, 64.0 * experiment_epsilon);
    if (assigned) return "AssignedDataOracleMismatch";
  }
  for (std::size_t p_index = 0; p_index < evidence.convergence.size(); ++p_index)
    for (std::size_t mesh_index = 0; mesh_index < 3; ++mesh_index)
    {
      const SolvedMeshEvidence& solved =
          evidence.convergence[p_index].solves[mesh_index];
      const ContinuousOracleResult& oracle = oracles.solves[p_index][mesh_index];
      if (oracle.volume.relative_l2 == 1.0e-10 &&
          solved.q_e_plus_four_volume.relative_l2 != 1.0e-10)
      {
        return "ErrorFloorOracleMismatch";
      }
      const double volume_difference = std::abs(
          solved.q_e_plus_four_volume.relative_l2 -
          oracle.volume.relative_l2) /
          std::max({std::abs(solved.q_e_plus_four_volume.relative_l2),
                    std::abs(oracle.volume.relative_l2), 1.0e-300});
      const double trace_difference = std::abs(
          solved.q_e_plus_four_trace.relative_l2 - oracle.trace.relative_l2) /
          std::max({std::abs(solved.q_e_plus_four_trace.relative_l2),
                    std::abs(oracle.trace.relative_l2), 1.0e-300});
      if (volume_difference > 2.0e-8 || trace_difference > 2.0e-8)
      {
        return "QuadratureOracleMismatch";
      }
    }
  return {};
}

bool TestActiveEvidenceControls(const ExperimentEvidence& baseline,
                                const IndependentCampaignOracles& baseline_oracles)
{
  if (!ValidateEvidenceCandidate(baseline, baseline_oracles).empty()) return false;
  {
    ExperimentEvidence candidate = baseline;
    candidate.rollback.restored_interior[0] += 1.0e-12;
    if (ValidateEvidenceCandidate(candidate, baseline_oracles) !=
        "RollbackVectorMismatch") return false;
  }
  {
    ExperimentEvidence candidate = baseline;
    candidate.rollback.local_copy_rejected = false;
    if (ValidateEvidenceCandidate(candidate, baseline_oracles) !=
        "RollbackConsumeFlagMismatch") return false;
  }
  {
    ExperimentEvidence candidate = baseline;
    candidate.convergence[0].solves[0].final_interior =
        candidate.convergence[0].solves[0].initial_interior;
    candidate.convergence[0].solves[0].final_trace =
        candidate.convergence[0].solves[0].initial_trace;
    if (ValidateEvidenceCandidate(candidate, baseline_oracles) !=
        "NoUpdateFinalEqualsInitial") return false;
  }
  {
    ExperimentEvidence candidate = baseline;
    candidate.convergence[0].solves[0].q_e_plus_four_trace.error_squared *= 2.0;
    if (ValidateEvidenceCandidate(candidate, baseline_oracles) !=
        "SkeletonNumeratorMismatch") return false;
  }
  {
    ExperimentEvidence candidate = baseline;
    candidate.convergence[0].solves[0].q_e_plus_four_trace.exact_norm_squared *=
        2.0;
    if (ValidateEvidenceCandidate(candidate, baseline_oracles) !=
        "SkeletonDenominatorMismatch") return false;
  }
  {
    ExperimentEvidence candidate = baseline;
    candidate.convergence[0].solves[0].q_e_plus_four_trace.measure *= 2.0;
    if (ValidateEvidenceCandidate(candidate, baseline_oracles) !=
        "SkeletonMeasureMismatch") return false;
  }
  {
    ExperimentEvidence candidate = baseline;
    ConvergenceEvidence& p_two = candidate.convergence[1];
    for (std::size_t mesh_index = 0; mesh_index < 3; ++mesh_index)
    {
      const double assigned = 1.0e-3 * std::pow(
          3.0 / p_two.mesh_widths[mesh_index], 3.0);
      SetCoupledRelativeError(p_two.solves[mesh_index].q_e_volume, assigned);
      SetCoupledRelativeError(p_two.solves[mesh_index].q_e_plus_four_volume,
                              assigned);
      SetCoupledRelativeError(p_two.solves[mesh_index].q_e_trace, assigned);
      SetCoupledRelativeError(p_two.solves[mesh_index].q_e_plus_four_trace,
                              assigned);
      p_two.volume_relative_l2[mesh_index] = assigned;
      p_two.trace_relative_l2[mesh_index] = assigned;
    }
    p_two.volume_rates = {{3.0, 3.0}};
    p_two.trace_rates = {{3.0, 3.0}};
    if (ValidateEvidenceCandidate(candidate, baseline_oracles) !=
        "AssignedDataOracleMismatch") return false;
  }
  {
    ExperimentEvidence candidate = baseline;
    ErrorIntegralEvidence& changed =
        candidate.convergence[0].solves[0].q_e_plus_four_volume;
    changed.error_squared *= 1.125 * 1.125;
    changed.relative_l2 *= 1.125;
    changed.direct_l2 *= 1.125;
    changed.rms *= 1.125;
    if (ValidateEvidenceCandidate(candidate, baseline_oracles) !=
        "QuadratureOracleMismatch") return false;
  }
  {
    ExperimentEvidence candidate = baseline;
    IndependentCampaignOracles oracles = baseline_oracles;
    ErrorIntegralEvidence& q_e = candidate.convergence[0].solves[0].q_e_volume;
    ErrorIntegralEvidence& q_e_plus_four =
        candidate.convergence[0].solves[0].q_e_plus_four_volume;
    SetCoupledRelativeError(q_e, 1.0e-10);
    SetCoupledRelativeError(q_e_plus_four, 1.0e-10);
    SetCoupledRelativeError(oracles.solves[0][0].volume, 1.0e-10);
    q_e_plus_four.error_squared *= 1.125 * 1.125;
    q_e_plus_four.relative_l2 *= 1.125;
    q_e_plus_four.direct_l2 *= 1.125;
    q_e_plus_four.rms *= 1.125;
    if (ValidateEvidenceCandidate(candidate, oracles) !=
        "ErrorFloorOracleMismatch") return false;
  }
  {
    ExperimentEvidence candidate = baseline;
    if (candidate.convergence.size() < 2 ||
        candidate.convergence[1].solves[0].polynomial_order != 2 ||
        candidate.convergence[1].solves[0].elements_per_direction != 3)
    {
      return false;
    }
    TrialAuditSnapshot* accepted = nullptr;
    for (TrialAuditSnapshot& audit :
         candidate.convergence[1].solves[0].trial_audits)
      if (audit.accepted)
      {
        if (accepted != nullptr) return false;
        accepted = &audit;
      }
    if (accepted == nullptr || !accepted->admissibility.passed ||
        !FiniteVector(accepted->trial_interior) ||
        !FiniteVector(accepted->trial_trace) ||
        !std::isfinite(accepted->alpha) ||
        !std::isfinite(accepted->base_residual))
    {
      return false;
    }
    accepted->armijo_boundary =
        (1.0 - 1.0e-4 * accepted->alpha) * accepted->base_residual;
    const double positive_margin = 1024.0 * experiment_epsilon *
        std::max(1.0, std::abs(accepted->armijo_boundary));
    accepted->trial_residual = accepted->armijo_boundary + positive_margin;
    if (!std::isfinite(accepted->trial_residual) ||
        !(accepted->trial_residual > accepted->armijo_boundary) ||
        ValidateEvidenceCandidate(candidate, baseline_oracles) !=
        "ArmijoBoundaryViolation") return false;
  }
  {
    ExperimentEvidence candidate = baseline;
    candidate.convergence[0].solves[0].trial_history.clear();
    if (ValidateEvidenceCandidate(candidate, baseline_oracles) !=
        "ArmijoRecordCountMismatch") return false;
  }
  return true;
}

int TestCompleteExperimentGate()
{
  ExperimentEvidence evidence{};
  if (!RunSmoothEulerExperiment(evidence) || !evidence.passed ||
      evidence.status != ExperimentStatus::Success) return 1;
  if (!evidence.rollback.passed || evidence.rollback.returned_success ||
      evidence.rollback.status != ExperimentStatus::IterationLimit ||
      evidence.rollback.accepted_updates != 1 ||
      !std::isfinite(evidence.rollback.initial_residual) ||
      !std::isfinite(evidence.rollback.residual_after_first_accepted) ||
      !evidence.rollback.residual_after_first_above_convergence ||
      !(evidence.rollback.residual_after_first_accepted >
        1.0e-10 * (1.0 + evidence.rollback.initial_residual)) ||
      !evidence.rollback.interior_byte_equal ||
      !evidence.rollback.trace_byte_equal) return 2;
  IndependentContext rollback_context;
  std::vector<double> rollback_old_interior, rollback_old_trace,
      rollback_initial_interior, rollback_initial_trace;
  OracleAdmissibility rollback_admissibility{};
  if (!BuildIndependentContext(1, 3, rollback_context) ||
      !IndependentProjection(rollback_context, false, rollback_old_interior,
                             rollback_old_trace) ||
      !IndependentProjection(rollback_context, true, rollback_initial_interior,
                             rollback_initial_trace) ||
      !FiniteVector(rollback_old_interior) || !FiniteVector(rollback_old_trace) ||
      !FiniteVector(rollback_initial_interior) ||
      !FiniteVector(rollback_initial_trace) ||
      !FiniteVector(evidence.rollback.entry_interior) ||
      !FiniteVector(evidence.rollback.entry_trace) ||
      !FiniteVector(evidence.rollback.restored_interior) ||
      !FiniteVector(evidence.rollback.restored_trace) ||
      !VectorWithinScale(rollback_initial_interior,
                         evidence.rollback.entry_interior, 4096.0) ||
      !VectorWithinScale(rollback_initial_trace, evidence.rollback.entry_trace,
                         4096.0) ||
      !IndependentPrescreen(rollback_context, 3 * 1 + 8,
                            evidence.rollback.entry_interior,
                            evidence.rollback.entry_trace,
                            rollback_admissibility) ||
      !rollback_admissibility.passed) return 3;
  std::vector<double> rollback_r, rollback_g;
  std::vector<int> rollback_map;
  if (!AssembleIndependentGlobalQ(
          rollback_context, 3 * 1 + 8, rollback_old_interior,
          evidence.rollback.entry_interior, evidence.rollback.entry_trace,
          rollback_r, rollback_g, rollback_map)) return 3;
  const double rollback_initial_residual = IndependentResidualNorm(
      rollback_r, rollback_g, rollback_map,
      rollback_context.trace_space->GetVSize());
  if (!Close(rollback_initial_residual, evidence.rollback.initial_residual,
             4096.0 * experiment_epsilon)) return 3;
  if (evidence.convergence.size() != 4) return 4;
  int solve_count = 0;
  for (std::size_t p_index = 0; p_index < evidence.convergence.size(); ++p_index)
  {
    const ConvergenceEvidence& ladder = evidence.convergence[p_index];
    if (ladder.polynomial_order != static_cast<int>(p_index) + 1 ||
        ladder.mesh_widths != std::array<int, 3>{{3, 6, 12}} ||
        ladder.status != ExperimentStatus::Success || !ladder.passed)
    {
      return 5;
    }
    for (std::size_t mesh_index = 0; mesh_index < 3; ++mesh_index)
    {
      const SolvedMeshEvidence& solved = ladder.solves[mesh_index];
      ++solve_count;
      if (solved.status != ExperimentStatus::Success ||
          solved.polynomial_order != ladder.polynomial_order ||
          solved.elements_per_direction != ladder.mesh_widths[mesh_index] ||
          solved.q_a != 3 * solved.polynomial_order + 8 ||
          solved.q_e != 4 * solved.polynomial_order + 12 ||
          solved.q_e_plus_four != solved.q_e + 4 ||
          !solved.internal_consistency) return 6;
      IndependentContext context;
      if (!BuildIndependentContext(solved.polynomial_order,
                                   solved.elements_per_direction, context)) return 7;
      if (!CheckProjectionAndInitialAdmissibility(context, solved)) return 8;
      if (!CheckLocalAssemblyAndAllColumns(context, solved)) return 9;
      if (!CheckGlobalResidualsAndTrials(context, solved)) return 10;
      if (!CheckLinearAudit(solved)) return 11;
    }
    const double minimum_volume = ladder.polynomial_order + 0.25;
    const double minimum_trace = ladder.polynomial_order + 0.50;
    for (int interval = 0; interval < 2; ++interval)
    {
      if (ladder.volume_rates[interval] < minimum_volume ||
          ladder.trace_rates[interval] < minimum_trace) return 12;
    }
  }
  if (solve_count != 12) return 13;
  IndependentCampaignOracles oracles{};
  if (!CheckContinuousErrors(evidence, oracles)) return 14;
  if (!TestActiveEvidenceControls(evidence, oracles)) return 15;
  return 0;
}

} // namespace

int main()
{
  const int flux = TestHandFluxAndFailureAtomicity();
  if (flux != 0)
  {
    std::cerr << "FAIL group=TestHandFluxAndFailureAtomicity subcode=" << flux
              << " exit=" << 10 + flux << '\n';
    return 10 + flux;
  }
  const int jacobian = TestAnalyticFluxJacobian();
  if (jacobian != 0)
  {
    std::cerr << "FAIL group=TestAnalyticFluxJacobian subcode=" << jacobian
              << " exit=" << 20 + jacobian << '\n';
    return 20 + jacobian;
  }
  const int source = TestManufacturedSource();
  if (source != 0)
  {
    std::cerr << "FAIL group=TestManufacturedSource subcode=" << source
              << " exit=" << 30 + source << '\n';
    return 30 + source;
  }
  const int ordering = TestMfemByNodesAndSignedVDofs();
  if (ordering != 0)
  {
    std::cerr << "FAIL group=TestMfemByNodesAndSignedVDofs subcode=" << ordering
              << " exit=" << 40 + ordering << '\n';
    return 40 + ordering;
  }
  const int trace_ordering = TestProjectedTraceComponentOrdering();
  if (trace_ordering != 0)
  {
    std::cerr << "FAIL group=TestProjectedTraceComponentOrdering subcode="
              << trace_ordering << " exit=" << 50 + trace_ordering << '\n';
    return 50 + trace_ordering;
  }
  const int orientation = TestElementFaceOrientationAndPermutation();
  if (orientation != 0)
  {
    std::cerr << "FAIL group=TestElementFaceOrientationAndPermutation subcode="
              << orientation << " exit=" << 60 + orientation << '\n';
    return 60 + orientation;
  }
  const int setup = TestFailurePreservingSetup();
  if (setup != 0)
  {
    std::cerr << "FAIL group=TestFailurePreservingSetup subcode=" << setup
              << " exit=" << 70 + setup << '\n';
    return 70 + setup;
  }
  const int face_oracles = TestFixedScalarTauConstantAndConservationOracles();
  if (face_oracles != 0)
  {
    std::cerr << "FAIL group=TestFixedScalarTauConstantAndConservationOracles subcode="
              << face_oracles << " exit=" << 80 + face_oracles << '\n';
    return 80 + face_oracles;
  }
  const int snapshots = TestSnapshotGenerationAndExtentValidation();
  if (snapshots != 0)
  {
    std::cerr << "FAIL group=TestSnapshotGenerationAndExtentValidation subcode="
              << snapshots << " exit=" << 90 + snapshots << '\n';
    return 90 + snapshots;
  }
  const int assembled_oracle = TestIndependentAssembledLinearizationOracle();
  if (assembled_oracle != 0)
  {
    std::cerr << "FAIL group=TestIndependentAssembledLinearizationOracle subcode="
              << assembled_oracle << " exit=" << 110 + assembled_oracle << '\n';
    return 110 + assembled_oracle;
  }
  const int two_sided_orientation = TestTwoSidedP2OrientationOracle();
  if (two_sided_orientation != 0)
  {
    std::cerr << "FAIL group=TestTwoSidedP2OrientationOracle subcode="
              << two_sided_orientation << " exit="
              << 130 + two_sided_orientation << '\n';
    return 130 + two_sided_orientation;
  }
  const int block_oracles = TestFaceBlockDirectionalAndIndependentSchurOracles();
  if (block_oracles != 0)
  {
    std::cerr << "FAIL group=TestFaceBlockDirectionalAndIndependentSchurOracles subcode="
              << block_oracles << " exit=" << 150 + block_oracles << '\n';
    return 150 + block_oracles;
  }
  const int experiment = TestCompleteExperimentGate();
  if (experiment != 0)
  {
    std::cerr << "FAIL group=TestCompleteExperimentGate subcode=" << experiment
              << " exit=" << 180 + experiment << '\n';
  }
  return experiment == 0 ? 0 : 180 + experiment;
}

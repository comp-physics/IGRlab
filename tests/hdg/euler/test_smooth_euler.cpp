#include "numerics/hdg/euler/smooth_euler.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
using dg::hdg::euler::LocalLinearizationSnapshot;
using dg::hdg::euler::backward_euler_dt;
using dg::hdg::euler::dimension;
using dg::hdg::euler::fixed_scalar_tau;
using dg::hdg::euler::conserved_components;
using dg::hdg::euler::gamma;

bool Close(const double left, const double right, const double tolerance)
{
  return std::abs(left - right) <= tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

double InfinityNorm(const std::vector<double>& values)
{
  double result = 0.0;
  for (const double value : values) { result = std::max(result, std::abs(value)); }
  return result;
}

double DifferenceNorm(const std::vector<double>& left,
                      const std::vector<double>& right)
{
  if (left.size() != right.size()) return std::numeric_limits<double>::infinity();
  double result = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i)
  {
    result = std::max(result, std::abs(left[i] - right[i]));
  }
  return result;
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
  int element_one_axis = -1;
  int element_two_axis = -1;
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

bool AssembleIndependentQ(const SmoothEulerExperiment& experiment,
                          const int element, const std::vector<double>& interior,
                          const std::vector<double>& base_interior,
                          const std::vector<double>& trace,
                          std::vector<double>& interior_residual,
                          std::vector<double>& trace_residual,
                          const OrientationCase* orientation = nullptr,
                          const OrientationMutation mutation = OrientationMutation::none)
{
  // Reintegrate nonlinear Q directly from coefficients, FE shapes, geometry,
  // and explicit quadrature.  No production residual/Jacobian/condensation,
  // snapshot-population, or orientation helper is called here.
  const mfem::GridFunction* interior_field = experiment.old_interior();
  const mfem::FiniteElementSpace* trace_space = experiment.trace_space();
  if (interior_field == nullptr || interior_field->FESpace() == nullptr ||
      trace_space == nullptr || trace_space->GetMesh() == nullptr) return false;
  const mfem::FiniteElementSpace& interior_space = *interior_field->FESpace();
  mfem::Mesh& mesh = *trace_space->GetMesh();
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
      mfem::IntRules.Get(fe->GetGeomType(), experiment.quadrature_order());
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
    const mfem::FiniteElement* trace_fe = trace_space->GetFaceElement(face);
    mfem::Array<int> face_vdofs;
    trace_space->GetFaceVDofs(face, face_vdofs);
    if (face_transform == nullptr || face_transform->Face == nullptr ||
        trace_fe == nullptr ||
        face_vdofs.Size() != conserved_components * trace_fe->GetDof() ||
        trace_offset + face_vdofs.Size() > static_cast<int>(trace.size())) return false;
    const bool is_element_one = face_transform->Elem1No == element;
    if (!is_element_one && face_transform->Elem2No != element) return false;
    const int trace_scalar_dofs = trace_fe->GetDof();
    const mfem::IntegrationRule& face_rule =
        mfem::IntRules.Get(trace_fe->GetGeomType(), experiment.quadrature_order());
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
      if (explicit_orientation)
      {
        const int axis = is_element_one ? orientation->element_one_axis :
                                          orientation->element_two_axis;
        if (axis < 0 || axis >= dimension) return false;
        mfem::IntegrationPoint side_point;
        side_point.x = axis == 0 ? element_point.x : element_point.y;
        trace_fe->CalcShape(side_point, trace_shape);
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
          int canonical_node = node;
          if (explicit_orientation && !is_element_one &&
              mutation != OrientationMutation::wrong_elem2_permutation)
          {
            canonical_node = trace_scalar_dofs - 1 - node;
          }
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
          int canonical_row = row;
          if (explicit_orientation && !is_element_one &&
              mutation != OrientationMutation::wrong_elem2_permutation)
          {
            canonical_row = trace_scalar_dofs - 1 - row;
          }
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
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    mfem::FaceElementTransformations* transform =
        mesh.GetFaceElementTransformations(face);
    const mfem::FiniteElement* face_fe = trace_space->GetFaceElement(face);
    if (transform == nullptr || transform->Face == nullptr || face_fe == nullptr ||
        transform->Elem1No < 0 || transform->Elem2No < 0 ||
        transform->Elem1No == transform->Elem2No || face_fe->GetDof() != 3)
    {
      continue;
    }
    mfem::IntegrationPoint endpoint_zero, endpoint_one, midpoint;
    endpoint_zero.x = 0.0;
    endpoint_one.x = 1.0;
    midpoint.x = 0.5;
    transform->SetAllIntPoints(&endpoint_zero);
    const mfem::IntegrationPoint element_one_zero =
        transform->GetElement1IntPoint();
    const mfem::IntegrationPoint element_two_zero =
        transform->GetElement2IntPoint();
    transform->SetAllIntPoints(&endpoint_one);
    const mfem::IntegrationPoint element_one_one =
        transform->GetElement1IntPoint();
    const mfem::IntegrationPoint element_two_one =
        transform->GetElement2IntPoint();
    const double one_delta[2] = {element_one_one.x - element_one_zero.x,
                                 element_one_one.y - element_one_zero.y};
    const double two_delta[2] = {element_two_one.x - element_two_zero.x,
                                 element_two_one.y - element_two_zero.y};
    const int one_axis = std::abs(one_delta[0]) > std::abs(one_delta[1]) ? 0 : 1;
    const int two_axis = std::abs(two_delta[0]) > std::abs(two_delta[1]) ? 0 : 1;
    if (!(one_delta[one_axis] > 0.5 && two_delta[two_axis] < -0.5)) continue;

    transform->SetAllIntPoints(&midpoint);
    mfem::Vector face_physical(dimension), one_physical(dimension),
        two_physical(dimension);
    transform->Face->Transform(midpoint, face_physical);
    mesh.GetElementTransformation(transform->Elem1No)->Transform(
        transform->GetElement1IntPoint(), one_physical);
    mesh.GetElementTransformation(transform->Elem2No)->Transform(
        transform->GetElement2IntPoint(), two_physical);
    // Both midpoint coordinates strictly inside the unit square exclude the
    // periodic seams; the generated Cartesian mesh is ordinary conforming.
    if (!(face_physical[0] > 1.0e-12 && face_physical[0] < 1.0 - 1.0e-12 &&
          face_physical[1] > 1.0e-12 && face_physical[1] < 1.0 - 1.0e-12))
    {
      continue;
    }
    bool coincident = true;
    const mfem::IntegrationRule& coincidence_rule = mfem::IntRules.Get(
        face_fe->GetGeomType(), experiment.quadrature_order());
    for (int node = 0; node < coincidence_rule.GetNPoints(); ++node)
    {
      const mfem::IntegrationPoint& point = coincidence_rule.IntPoint(node);
      transform->SetAllIntPoints(&point);
      transform->Face->Transform(point, face_physical);
      mesh.GetElementTransformation(transform->Elem1No)->Transform(
          transform->GetElement1IntPoint(), one_physical);
      mesh.GetElementTransformation(transform->Elem2No)->Transform(
          transform->GetElement2IntPoint(), two_physical);
      for (int coordinate = 0; coordinate < dimension; ++coordinate)
      {
        coincident = coincident &&
            std::abs(face_physical[coordinate] - one_physical[coordinate]) <= 2.0e-14 &&
            std::abs(face_physical[coordinate] - two_physical[coordinate]) <= 2.0e-14;
      }
    }
    if (!coincident) continue;
    selected.face = face;
    selected.element_one = transform->Elem1No;
    selected.element_two = transform->Elem2No;
    selected.element_one_axis = one_axis;
    selected.element_two_axis = two_axis;
    return true;
  }
  return false;
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
        const mfem::IntegrationPoint& element_point = transform->Elem1No == element ?
            transform->GetElement1IntPoint() : transform->GetElement2IntPoint();
        mfem::Vector face_physical(dg::hdg::euler::dimension);
        mfem::Vector element_physical(dg::hdg::euler::dimension);
        transform->Face->Transform(face_point, face_physical);
        mesh->GetElementTransformation(element)->Transform(element_point, element_physical);
        if (!Close(face_physical[0], element_physical[0], 2.0e-14) ||
            !Close(face_physical[1], element_physical[1], 2.0e-14)) return 5;
      }
      offset += face_vdofs.Size();
    }
    if (offset != snapshot.trace_dofs || orientations.Size() != faces.Size()) return 6;
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
  constexpr double epsilon = std::numeric_limits<double>::epsilon();
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
  constexpr double epsilon = std::numeric_limits<double>::epsilon();
  double elem2_correct_error = 0.0;
  double elem2_wrong_normal_error = 0.0;
  double elem2_wrong_permutation_error = 0.0;
  const int elements[2] = {orientation.element_one, orientation.element_two};
  for (const int element : elements)
  {
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
    if (trace_direction[trace_offset] == trace_direction[trace_offset + 1] ||
        trace_direction[trace_offset + 1] == trace_direction[trace_offset + 2] ||
        trace_direction[trace_offset] == trace_direction[trace_offset + 2] ||
        InfinityNorm(trace_direction) != 1.0) return 7;
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
    if (!std::isfinite(correct_error) ||
        DifferenceNorm(correct_r, b_action) > trace_tolerance ||
        DifferenceNorm(correct_g, d_action) > trace_tolerance) return 12;
    if (element == orientation.element_two)
    {
      elem2_correct_error = correct_error;
      std::vector<double> mutated_r, mutated_g;
      elem2_wrong_normal_error = trace_action_error(
          OrientationMutation::wrong_elem2_normal, mutated_r, mutated_g);
      elem2_wrong_permutation_error = trace_action_error(
          OrientationMutation::wrong_elem2_permutation, mutated_r, mutated_g);
    }
  }
  const double ratio_floor = 100.0 * std::max(elem2_correct_error, epsilon);
  if (!std::isfinite(elem2_wrong_normal_error) ||
      elem2_wrong_normal_error < 1.0e-5 ||
      elem2_wrong_normal_error < ratio_floor) return 13;
  if (!std::isfinite(elem2_wrong_permutation_error) ||
      elem2_wrong_permutation_error < 1.0e-5 ||
      elem2_wrong_permutation_error < ratio_floor) return 14;
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

int TestCompleteExperimentGate()
{
  ExperimentEvidence evidence{};
  evidence.passed = true;
  // This is a real complete-gate invocation, not a print-only status.  03B
  // ends at a fixed-state linearization; full Newton/convergence remains
  // deliberately unavailable until 03C, so the top-level experiment is an
  // explicit fail-closed gate rather than a false success.
  return !RunSmoothEulerExperiment(evidence) && !evidence.passed ? 0 : 1;
}

} // namespace

int main()
{
  const int flux = TestHandFluxAndFailureAtomicity();
  if (flux != 0) { return 10 + flux; }
  const int jacobian = TestAnalyticFluxJacobian();
  if (jacobian != 0) { return 20 + jacobian; }
  const int source = TestManufacturedSource();
  if (source != 0) { return 30 + source; }
  const int ordering = TestMfemByNodesAndSignedVDofs();
  if (ordering != 0) { return 40 + ordering; }
  const int trace_ordering = TestProjectedTraceComponentOrdering();
  if (trace_ordering != 0) { return 50 + trace_ordering; }
  const int orientation = TestElementFaceOrientationAndPermutation();
  if (orientation != 0) { return 60 + orientation; }
  const int setup = TestFailurePreservingSetup();
  if (setup != 0) { return 70 + setup; }
  const int face_oracles = TestFixedScalarTauConstantAndConservationOracles();
  if (face_oracles != 0) { return 80 + face_oracles; }
  const int snapshots = TestSnapshotGenerationAndExtentValidation();
  if (snapshots != 0) { return 90 + snapshots; }
  const int assembled_oracle = TestIndependentAssembledLinearizationOracle();
  if (assembled_oracle != 0) { return 110 + assembled_oracle; }
  const int two_sided_orientation = TestTwoSidedP2OrientationOracle();
  if (two_sided_orientation != 0) { return 130 + two_sided_orientation; }
  const int block_oracles = TestFaceBlockDirectionalAndIndependentSchurOracles();
  if (block_oracles != 0) { return 150 + block_oracles; }
  const int experiment = TestCompleteExperimentGate();
  return experiment == 0 ? 0 : 180 + experiment;
}

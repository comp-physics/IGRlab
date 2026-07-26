#include "numerics/hdg/euler/smooth_euler_mpi.hpp"

#include "numerics/hdg/core/host_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace dg::hdg::euler
{
namespace
{

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double armijo_constant = 1.0e-4;
constexpr int maximum_line_search_trials = 13;

template <typename T>
StridedBlockView<T> View(T* data, const LocalIndex batches,
                         const LocalIndex rows,
                         const LocalIndex columns) noexcept
{
  return {data, batches, rows, columns, rows * columns, columns, 1};
}

bool IsFinite(const double value) noexcept { return std::isfinite(value); }

bool FiniteValues(const std::vector<double>& values) noexcept
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return IsFinite(value); });
}

std::vector<double> VectorValues(const mfem::Vector& vector)
{
  const double* const data = vector.HostRead();
  if (data == nullptr && vector.Size() != 0)
  {
    throw std::runtime_error("MFEM vector has no readable host storage");
  }
  return data == nullptr ? std::vector<double>{} :
      std::vector<double>(data, data + vector.Size());
}

bool AssignValues(mfem::Vector& vector, const std::vector<double>& values)
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

bool InterpolateByNodes(const mfem::Vector& field,
                        const mfem::Array<int>& signed_vdofs,
                        const mfem::Vector& shape,
                        ConservedState& state)
{
  if (signed_vdofs.Size() != conserved_components * shape.Size())
  {
    return false;
  }
  state = {};
  const double* const values = field.HostRead();
  if (values == nullptr) { return false; }
  for (int scalar = 0; scalar < shape.Size(); ++scalar)
    for (int component = 0; component < conserved_components; ++component)
    {
      double sign = 1.0;
      const int vdof = DecodeSignedVDof(
          signed_vdofs[ByNodesIndex(scalar, shape.Size(), component)], sign);
      if (vdof < 0 || vdof >= field.Size()) { return false; }
      state[component] += shape[scalar] * sign * values[vdof];
    }
  return std::all_of(state.begin(), state.end(),
                     [](const double value) { return IsFinite(value); });
}

std::vector<int> BuildPartition(const mfem::Mesh& mesh,
                                const int rank_count)
{
  std::vector<int> partition(static_cast<std::size_t>(mesh.GetNE()), 0);
  if (rank_count == 1) { return partition; }
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    partition[static_cast<std::size_t>(element)] =
        element * rank_count / mesh.GetNE();
  }
  return partition;
}

struct LocalSystem
{
  int elements = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  std::vector<double> residual_interior;
  std::vector<double> residual_trace;
  std::vector<double> a, b, c, d, f, g;
};

bool HasLocalExtents(const LocalSystem& system)
{
  if (system.elements <= 0 || system.interior_dofs <= 0 ||
      system.trace_dofs <= 0)
  {
    return false;
  }
  const std::size_t elements = static_cast<std::size_t>(system.elements);
  const std::size_t ni = static_cast<std::size_t>(system.interior_dofs);
  const std::size_t nt = static_cast<std::size_t>(system.trace_dofs);
  return system.residual_interior.size() == elements * ni &&
      system.residual_trace.size() == elements * nt &&
      system.a.size() == elements * ni * ni &&
      system.b.size() == elements * ni * nt &&
      system.c.size() == elements * nt * ni &&
      system.d.size() == elements * nt * nt &&
      system.f.size() == elements * ni &&
      system.g.size() == elements * nt &&
      FiniteValues(system.residual_interior) &&
      FiniteValues(system.residual_trace) && FiniteValues(system.a) &&
      FiniteValues(system.b) && FiniteValues(system.c) &&
      FiniteValues(system.d) && FiniteValues(system.f) &&
      FiniteValues(system.g);
}

bool AssembleLocalSystem(
    mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& interior_space,
    mfem::ParFiniteElementSpace& trace_space,
    const mfem::ParGridFunction& old_interior,
    const mfem::ParGridFunction& current_interior,
    const mfem::ParGridFunction& current_trace,
    const int quadrature_order, const int local_interior,
    const int local_trace, LocalSystem& system)
{
  LocalSystem candidate;
  candidate.elements = mesh.GetNE();
  candidate.interior_dofs = local_interior;
  candidate.trace_dofs = local_trace;
  if (candidate.elements <= 0 || local_interior <= 0 || local_trace <= 0)
  {
    return false;
  }
  const std::size_t batches = static_cast<std::size_t>(candidate.elements);
  const std::size_t ni = static_cast<std::size_t>(local_interior);
  const std::size_t nt = static_cast<std::size_t>(local_trace);
  candidate.residual_interior.assign(batches * ni, 0.0);
  candidate.residual_trace.assign(batches * nt, 0.0);
  candidate.a.assign(batches * ni * ni, 0.0);
  candidate.b.assign(batches * ni * nt, 0.0);
  candidate.c.assign(batches * nt * ni, 0.0);
  candidate.d.assign(batches * nt * nt, 0.0);

  for (int element = 0; element < candidate.elements; ++element)
  {
    const mfem::FiniteElement* const finite_element =
        interior_space.GetFE(element);
    mfem::ElementTransformation* const transformation =
        mesh.GetElementTransformation(element);
    if (finite_element == nullptr || transformation == nullptr) { return false; }
    const int scalar_dofs = finite_element->GetDof();
    mfem::Array<int> interior_vdofs;
    interior_space.GetElementVDofs(element, interior_vdofs);
    if (scalar_dofs <= 0 || interior_vdofs.Size() != local_interior)
    {
      return false;
    }
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(
        finite_element->GetGeomType(), quadrature_order);
    const std::size_t a_offset = static_cast<std::size_t>(element) * ni * ni;
    for (int point_index = 0; point_index < rule.GetNPoints(); ++point_index)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(point_index);
      transformation->SetIntPoint(&point);
      mfem::Vector physical(dimension), shape(scalar_dofs);
      mfem::DenseMatrix derivative(scalar_dofs, dimension);
      transformation->Transform(point, physical);
      finite_element->CalcShape(point, shape);
      finite_element->CalcPhysDShape(*transformation, derivative);
      ConservedState state{}, old_state{}, source{};
      FluxVector flux_x{}, flux_y{};
      FluxJacobian jacobian_x{}, jacobian_y{};
      if (!InterpolateByNodes(current_interior, interior_vdofs, shape, state) ||
          !InterpolateByNodes(old_interior, interior_vdofs, shape, old_state) ||
          !ManufacturedSource(physical[0], physical[1], source).ok() ||
          !NormalFlux(state, {{1.0, 0.0}}, flux_x).ok() ||
          !NormalFlux(state, {{0.0, 1.0}}, flux_y).ok() ||
          !NormalFluxJacobian(state, {{1.0, 0.0}}, jacobian_x).ok() ||
          !NormalFluxJacobian(state, {{0.0, 1.0}}, jacobian_y).ok())
      {
        return false;
      }
      const double weight = point.weight * transformation->Weight();
      if (!IsFinite(weight) || !(weight > 0.0)) { return false; }
      for (int row = 0; row < scalar_dofs; ++row)
      {
        for (int row_component = 0;
             row_component < conserved_components; ++row_component)
        {
          const int local_row =
              ByNodesIndex(row, scalar_dofs, row_component);
          candidate.residual_interior[
              static_cast<std::size_t>(element) * ni + local_row] +=
              weight *
              (shape[row] * (state[row_component] - old_state[row_component]) /
                   backward_euler_dt -
               derivative(row, 0) * flux_x[row_component] -
               derivative(row, 1) * flux_y[row_component] -
               shape[row] * source[row_component]);
        }
        for (int column = 0; column < scalar_dofs; ++column)
          for (int row_component = 0;
               row_component < conserved_components; ++row_component)
            for (int column_component = 0;
                 column_component < conserved_components; ++column_component)
            {
              const int local_row =
                  ByNodesIndex(row, scalar_dofs, row_component);
              const int local_column =
                  ByNodesIndex(column, scalar_dofs, column_component);
              const double mass = row_component == column_component ?
                  shape[row] * shape[column] / backward_euler_dt : 0.0;
              const double volume =
                  -(derivative(row, 0) *
                        jacobian_x[row_component][column_component] +
                    derivative(row, 1) *
                        jacobian_y[row_component][column_component]) *
                  shape[column];
              candidate.a[a_offset + static_cast<std::size_t>(local_row) * ni +
                          local_column] += weight * (mass + volume);
            }
      }
    }

    mfem::Array<int> faces, orientations;
    mesh.GetElementEdges(element, faces, orientations);
    if (faces.Size() <= 0 || orientations.Size() != faces.Size()) { return false; }
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      mfem::FaceElementTransformations* const face_transformation =
          mesh.GetFaceElementTransformations(face);
      const mfem::FiniteElement* const trace_element =
          trace_space.GetFaceElement(face);
      mfem::Array<int> face_vdofs;
      trace_space.GetFaceVDofs(face, face_vdofs);
      if (face_transformation == nullptr ||
          face_transformation->Face == nullptr || trace_element == nullptr ||
          face_vdofs.Size() !=
              conserved_components * trace_element->GetDof())
      {
        return false;
      }
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
        const double face_measure = scaled_normal.Norml2();
        if (!IsFinite(face_measure) || !(face_measure > 0.0)) { return false; }
        const double side = element_one ? 1.0 : -1.0;
        const std::array<double, dimension> unit_normal{{
            side * scaled_normal[0] / face_measure,
            side * scaled_normal[1] / face_measure}};
        finite_element->CalcShape(element_point, interior_shape);
        trace_element->CalcShape(face_point, trace_shape);
        ConservedState interior_state{}, trace_state{};
        FluxVector trace_flux{};
        FluxJacobian trace_jacobian{};
        if (!InterpolateByNodes(current_interior, interior_vdofs,
                                interior_shape, interior_state) ||
            !InterpolateByNodes(current_trace, face_vdofs, trace_shape,
                                trace_state) ||
            !NormalFlux(trace_state, unit_normal, trace_flux).ok() ||
            !NormalFluxJacobian(trace_state, unit_normal,
                                trace_jacobian).ok())
        {
          return false;
        }
        const double weight = face_point.weight * face_measure;
        for (int row = 0; row < scalar_dofs; ++row)
          for (int component = 0; component < conserved_components;
               ++component)
          {
            const int local_row =
                ByNodesIndex(row, scalar_dofs, component);
            candidate.residual_interior[
                static_cast<std::size_t>(element) * ni + local_row] +=
                weight * interior_shape[row] *
                (trace_flux[component] + fixed_scalar_tau *
                     (interior_state[component] - trace_state[component]));
          }
        for (int row = 0; row < trace_scalar_dofs; ++row)
          for (int component = 0; component < conserved_components;
               ++component)
          {
            const int local_row = trace_offset +
                ByNodesIndex(row, trace_scalar_dofs, component);
            candidate.residual_trace[
                static_cast<std::size_t>(element) * nt + local_row] +=
                weight * trace_shape[row] *
                (trace_flux[component] + fixed_scalar_tau *
                     (interior_state[component] - trace_state[component]));
          }
        for (int row = 0; row < scalar_dofs; ++row)
          for (int column = 0; column < scalar_dofs; ++column)
            for (int component = 0; component < conserved_components;
                 ++component)
            {
              const int local_row =
                  ByNodesIndex(row, scalar_dofs, component);
              const int local_column =
                  ByNodesIndex(column, scalar_dofs, component);
              candidate.a[a_offset + static_cast<std::size_t>(local_row) * ni +
                          local_column] +=
                  weight * interior_shape[row] * fixed_scalar_tau *
                  interior_shape[column];
            }
        for (int row = 0; row < scalar_dofs; ++row)
          for (int column = 0; column < trace_scalar_dofs; ++column)
            for (int row_component = 0;
                 row_component < conserved_components; ++row_component)
              for (int column_component = 0;
                   column_component < conserved_components;
                   ++column_component)
              {
                const int local_row =
                    ByNodesIndex(row, scalar_dofs, row_component);
                const int local_column = trace_offset + ByNodesIndex(
                    column, trace_scalar_dofs, column_component);
                candidate.b[static_cast<std::size_t>(element) * ni * nt +
                            static_cast<std::size_t>(local_row) * nt +
                            local_column] +=
                    weight * interior_shape[row] * trace_shape[column] *
                    (trace_jacobian[row_component][column_component] -
                     (row_component == column_component ? fixed_scalar_tau :
                                                           0.0));
              }
        for (int row = 0; row < trace_scalar_dofs; ++row)
          for (int column = 0; column < scalar_dofs; ++column)
            for (int component = 0; component < conserved_components;
                 ++component)
            {
              const int local_row = trace_offset +
                  ByNodesIndex(row, trace_scalar_dofs, component);
              const int local_column =
                  ByNodesIndex(column, scalar_dofs, component);
              candidate.c[static_cast<std::size_t>(element) * nt * ni +
                          static_cast<std::size_t>(local_row) * ni +
                          local_column] +=
                  weight * trace_shape[row] * fixed_scalar_tau *
                  interior_shape[column];
            }
        for (int row = 0; row < trace_scalar_dofs; ++row)
          for (int column = 0; column < trace_scalar_dofs; ++column)
            for (int row_component = 0;
                 row_component < conserved_components; ++row_component)
              for (int column_component = 0;
                   column_component < conserved_components;
                   ++column_component)
              {
                const int local_row = trace_offset +
                    ByNodesIndex(row, trace_scalar_dofs, row_component);
                const int local_column = trace_offset + ByNodesIndex(
                    column, trace_scalar_dofs, column_component);
                candidate.d[static_cast<std::size_t>(element) * nt * nt +
                            static_cast<std::size_t>(local_row) * nt +
                            local_column] +=
                    weight * trace_shape[row] * trace_shape[column] *
                    (trace_jacobian[row_component][column_component] -
                     (row_component == column_component ? fixed_scalar_tau :
                                                           0.0));
              }
      }
      trace_offset += face_vdofs.Size();
    }
    if (trace_offset != local_trace) { return false; }
  }
  candidate.f.resize(candidate.residual_interior.size());
  candidate.g.resize(candidate.residual_trace.size());
  std::transform(candidate.residual_interior.begin(),
                 candidate.residual_interior.end(), candidate.f.begin(),
                 [](const double value) { return -value; });
  std::transform(candidate.residual_trace.begin(),
                 candidate.residual_trace.end(), candidate.g.begin(),
                 [](const double value) { return -value; });
  if (!HasLocalExtents(candidate)) { return false; }
  system = std::move(candidate);
  return true;
}

double GlobalResidualNorm(
    MPI_Comm communicator, mfem::ParFiniteElementSpace& trace_space,
    const LocalSystem& system, const std::vector<int>& signed_trace_dofs)
{
  const std::size_t expected_trace_dofs =
      static_cast<std::size_t>(system.elements) * system.trace_dofs;
  bool local_ok = system.elements > 0 && system.trace_dofs > 0 &&
      system.residual_trace.size() == expected_trace_dofs &&
      signed_trace_dofs.size() == expected_trace_dofs &&
      trace_space.GetVSize() > 0 && trace_space.GetTrueVSize() > 0 &&
      trace_space.Dof_TrueDof_Matrix() != nullptr;
  mfem::Vector local_trace_residual;
  mfem::Vector true_trace_residual;
  try
  {
    if (local_ok)
    {
      local_trace_residual.SetSize(trace_space.GetVSize());
      true_trace_residual.SetSize(trace_space.GetTrueVSize());
      local_trace_residual = 0.0;
      for (std::size_t local = 0; local < expected_trace_dofs; ++local)
      {
        double sign = 1.0;
        const int dof = DecodeSignedVDof(signed_trace_dofs[local], sign);
        local_ok = dof >= 0 && dof < local_trace_residual.Size();
        if (!local_ok) { break; }
        local_trace_residual[dof] += sign * system.residual_trace[local];
      }
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  int local_valid = local_ok ? 1 : 0;
  int global_valid = 0;
  MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN,
                communicator);
  if (global_valid != 1)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }

  try
  {
    trace_space.Dof_TrueDof_Matrix()->MultTranspose(local_trace_residual,
                                                    true_trace_residual);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  local_valid = local_ok ? 1 : 0;
  MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN,
                communicator);
  if (global_valid != 1)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double local_sum = 0.0;
  for (const double value : system.residual_interior)
  {
    local_sum += value * value;
  }
  for (int local = 0; local < true_trace_residual.Size(); ++local)
  {
    const double value = true_trace_residual[local];
    local_sum += value * value;
  }
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                communicator);
  return std::sqrt(global_sum);
}

bool PrescreenState(mfem::ParMesh& mesh,
                    mfem::ParFiniteElementSpace& interior_space,
                    mfem::ParFiniteElementSpace& trace_space,
                    const mfem::ParGridFunction& interior,
                    const mfem::ParGridFunction& trace,
                    const int quadrature_order, double& minimum_density,
                    double& minimum_pressure)
{
  double local_density = std::numeric_limits<double>::infinity();
  double local_pressure = std::numeric_limits<double>::infinity();
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    const mfem::FiniteElement* const finite_element =
        interior_space.GetFE(element);
    if (finite_element == nullptr) { return false; }
    mfem::Array<int> interior_vdofs;
    interior_space.GetElementVDofs(element, interior_vdofs);
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(
        finite_element->GetGeomType(), quadrature_order);
    mfem::Vector shape(finite_element->GetDof());
    for (int point_index = 0; point_index < rule.GetNPoints(); ++point_index)
    {
      finite_element->CalcShape(rule.IntPoint(point_index), shape);
      ConservedState state{};
      PrimitiveState primitive{};
      if (!InterpolateByNodes(interior, interior_vdofs, shape, state) ||
          !PrimitiveFromConservative(state, primitive).ok())
      {
        return false;
      }
      local_density = std::min(local_density, primitive[0]);
      local_pressure = std::min(local_pressure, primitive[3]);
    }
  }
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* const finite_element =
        trace_space.GetFaceElement(face);
    if (finite_element == nullptr) { return false; }
    mfem::Array<int> trace_vdofs;
    trace_space.GetFaceVDofs(face, trace_vdofs);
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(
        finite_element->GetGeomType(), quadrature_order);
    mfem::Vector shape(finite_element->GetDof());
    for (int point_index = 0; point_index < rule.GetNPoints(); ++point_index)
    {
      finite_element->CalcShape(rule.IntPoint(point_index), shape);
      ConservedState state{};
      PrimitiveState primitive{};
      if (!InterpolateByNodes(trace, trace_vdofs, shape, state) ||
          !PrimitiveFromConservative(state, primitive).ok())
      {
        return false;
      }
      local_density = std::min(local_density, primitive[0]);
      local_pressure = std::min(local_pressure, primitive[3]);
    }
  }
  if (!IsFinite(local_density) || !IsFinite(local_pressure) ||
      !(local_density > 0.0) || !(local_pressure > 0.0))
  {
    return false;
  }
  minimum_density = local_density;
  minimum_pressure = local_pressure;
  return true;
}

bool ProjectTrace(mfem::ParMesh& mesh,
                  mfem::ParFiniteElementSpace& trace_space,
                  mfem::ParGridFunction& trace, const bool perturbed)
{
  trace = 0.0;
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* const finite_element =
        trace_space.GetFaceElement(face);
    mfem::ElementTransformation* const transformation =
        mesh.GetFaceTransformation(face);
    if (finite_element == nullptr || transformation == nullptr) { return false; }
    const int scalar_dofs = finite_element->GetDof();
    mfem::Array<int> vdofs;
    trace_space.GetFaceVDofs(face, vdofs);
    if (scalar_dofs <= 0 ||
        vdofs.Size() != scalar_dofs * conserved_components)
    {
      return false;
    }
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    for (int scalar = 0; scalar < scalar_dofs; ++scalar)
    {
      mfem::Vector physical(dimension);
      transformation->Transform(nodes.IntPoint(scalar), physical);
      PrimitiveState primitive{};
      if (!ManufacturedPrimitive(physical[0], physical[1], primitive).ok())
      {
        return false;
      }
      if (perturbed)
      {
        const double psi_one = std::sin(2.0 * pi * physical[0]) *
            std::cos(2.0 * pi * physical[1]);
        const double psi_two = std::cos(2.0 * pi * physical[0]) *
            std::sin(2.0 * pi * physical[1]);
        const double psi_three =
            std::sin(2.0 * pi * (physical[0] + physical[1]));
        primitive[0] *= 1.0 + 0.04 * psi_one;
        primitive[1] += 0.03 * psi_two;
        primitive[2] -= 0.03 * psi_one;
        primitive[3] *= 1.0 + 0.04 * psi_three;
      }
      ConservedState state{};
      if (!ConservativeFromPrimitive(primitive, state).ok()) { return false; }
      for (int component = 0; component < conserved_components; ++component)
      {
        double sign = 1.0;
        const int vdof = DecodeSignedVDof(
            vdofs[ByNodesIndex(scalar, scalar_dofs, component)], sign);
        if (vdof < 0 || vdof >= trace.Size()) { return false; }
        trace[vdof] = sign * state[component];
      }
    }
  }
  return true;
}

} // namespace

struct SmoothEulerMPIExperiment::Implementation
{
  MPI_Comm communicator = MPI_COMM_NULL;
  int rank = -1;
  int ranks = 0;
  bool owns_communicator = false;

  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<mfem::L2_FECollection> interior_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> interior_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
  std::unique_ptr<mfem::ParGridFunction> old_interior;
  std::unique_ptr<mfem::ParGridFunction> old_trace;
  std::unique_ptr<mfem::ParGridFunction> current_interior;
  std::unique_ptr<mfem::ParGridFunction> current_trace;
  trace::DGInterfaceTraceExchangePlan exchange_plan;
  std::vector<int> signed_trace_dofs;
  int local_interior = 0;
  int local_trace = 0;
  int elements_per_direction = 0;
  SmoothEulerMPIStatus published_status = SmoothEulerMPIStatus::Uninitialized;
  std::uint64_t published_generation = 0;
  SmoothEulerMPIEvidence published_evidence{};
  SmoothEulerMPIFailurePoint last_failure_point_reached =
      SmoothEulerMPIFailurePoint::None;

  explicit Implementation(MPI_Comm source)
  {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0 || source == MPI_COMM_NULL ||
        MPI_Comm_dup(source, &communicator) != MPI_SUCCESS)
    {
      throw std::invalid_argument("smooth Euler MPI requires a live communicator");
    }
    owns_communicator = true;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &ranks);
  }

  ~Implementation()
  {
    // Destroy every MFEM object while the duplicated communicator is live.
    exchange_plan = {};
    current_trace.reset();
    current_interior.reset();
    old_trace.reset();
    old_interior.reset();
    trace_space.reset();
    interior_space.reset();
    trace_collection.reset();
    interior_collection.reset();
    mesh.reset();
    if (!owns_communicator || communicator == MPI_COMM_NULL) { return; }
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0) { MPI_Comm_free(&communicator); }
  }

  bool CollectiveAll(const bool local_value) const
  {
    int local = local_value ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, communicator);
    return global == 1;
  }

  bool Injected(const SmoothEulerMPIFailureInjection& injection,
                const SmoothEulerMPIFailurePoint point) const noexcept
  {
    return injection.point == point && injection.rank == rank;
  }

  bool AdmitPhase(const SmoothEulerMPIFailureInjection& injection,
                  const SmoothEulerMPIFailurePoint point)
  {
    const bool admitted = CollectiveAll(!Injected(injection, point));
    if (!admitted) { last_failure_point_reached = point; }
    return admitted;
  }
};

SmoothEulerMPIExperiment::SmoothEulerMPIExperiment(
    MPI_Comm communicator, const int polynomial_order)
    : implementation_(std::make_unique<Implementation>(communicator)),
      polynomial_order_(polynomial_order),
      quadrature_order_(0)
{
  const int local_valid = polynomial_order >= 1 && polynomial_order <= 4 ?
      1 : 0;
  int global_valid = 0;
  int minimum_order = 0;
  int maximum_order = 0;
  MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN,
                implementation_->communicator);
  MPI_Allreduce(&polynomial_order, &minimum_order, 1, MPI_INT, MPI_MIN,
                implementation_->communicator);
  MPI_Allreduce(&polynomial_order, &maximum_order, 1, MPI_INT, MPI_MAX,
                implementation_->communicator);
  if (global_valid != 1 || minimum_order != maximum_order)
  {
    throw std::invalid_argument(
        "smooth Euler MPI requires communicator-wide p in [1,4]");
  }
  quadrature_order_ = 3 * polynomial_order_ + 8;
}

SmoothEulerMPIExperiment::~SmoothEulerMPIExperiment() = default;
SmoothEulerMPIExperiment::SmoothEulerMPIExperiment(
    SmoothEulerMPIExperiment&&) noexcept = default;
SmoothEulerMPIExperiment& SmoothEulerMPIExperiment::operator=(
    SmoothEulerMPIExperiment&&) noexcept = default;

bool SmoothEulerMPIExperiment::Initialize(const int elements_per_direction)
{
  struct CandidateGraph
  {
    std::unique_ptr<mfem::ParMesh> mesh;
    std::unique_ptr<mfem::L2_FECollection> interior_collection;
    std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
    std::unique_ptr<mfem::ParFiniteElementSpace> interior_space;
    std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
    std::unique_ptr<mfem::ParGridFunction> old_interior;
    std::unique_ptr<mfem::ParGridFunction> old_trace;
    std::unique_ptr<mfem::ParGridFunction> current_interior;
    std::unique_ptr<mfem::ParGridFunction> current_trace;
    trace::DGInterfaceTraceExchangePlan exchange_plan;
    std::vector<int> signed_trace_dofs;
    int local_interior = 0;
    int local_trace = 0;
  } candidate;

  int minimum_elements = 0;
  int maximum_elements = 0;
  MPI_Allreduce(&elements_per_direction, &minimum_elements, 1, MPI_INT,
                MPI_MIN, implementation_->communicator);
  MPI_Allreduce(&elements_per_direction, &maximum_elements, 1, MPI_INT,
                MPI_MAX, implementation_->communicator);
  bool local_ok = elements_per_direction == 3 &&
      minimum_elements == maximum_elements &&
      (implementation_->ranks == 1 || implementation_->ranks == 2);
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  std::unique_ptr<mfem::Mesh> periodic;
  std::vector<int> partition;
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        elements_per_direction, elements_per_direction,
        mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(dimension));
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    periodic = std::make_unique<mfem::Mesh>(mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations)));
    partition = BuildPartition(*periodic, implementation_->ranks);
    candidate.interior_collection = std::make_unique<mfem::L2_FECollection>(
        polynomial_order_, dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    candidate.trace_collection =
        std::make_unique<mfem::DG_Interface_FECollection>(
            polynomial_order_, dimension, mfem::FiniteElement::VALUE,
            mfem::BasisType::GaussLegendre);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
    candidate.mesh = std::make_unique<mfem::ParMesh>(
        implementation_->communicator, *periodic, partition.data());
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
      candidate.interior_space =
          std::make_unique<mfem::ParFiniteElementSpace>(
              candidate.mesh.get(), candidate.interior_collection.get(),
              conserved_components, mfem::Ordering::byNODES);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
      candidate.trace_space =
          std::make_unique<mfem::ParFiniteElementSpace>(
              candidate.mesh.get(), candidate.trace_collection.get(),
              conserved_components, mfem::Ordering::byNODES);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
    local_ok = candidate.mesh != nullptr && candidate.mesh->GetNE() > 0 &&
        candidate.interior_space != nullptr && candidate.trace_space != nullptr;
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
    candidate.interior_space->ExchangeFaceNbrData();
    local_ok = candidate.interior_space->GetFaceNbrVSize() >= 0;
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
    candidate.trace_space->ExchangeFaceNbrData();
    local_ok = candidate.trace_space->GetFaceNbrVSize() >= 0;
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
      const mfem::FiniteElement* const interior_element =
          candidate.interior_space->GetFE(0);
      local_ok = interior_element != nullptr &&
          candidate.mesh->GetNE() > 0 &&
          candidate.trace_space->GetVSize() > 0;
      if (local_ok)
      {
        candidate.local_interior =
            conserved_components * interior_element->GetDof();
        mfem::Array<int> first_faces, first_orientations;
        candidate.mesh->GetElementEdges(0, first_faces, first_orientations);
        candidate.local_trace = 0;
        for (int local_face = 0; local_face < first_faces.Size(); ++local_face)
        {
          mfem::Array<int> vdofs;
          candidate.trace_space->GetFaceVDofs(first_faces[local_face], vdofs);
          candidate.local_trace += vdofs.Size();
        }
        local_ok = candidate.local_interior > 0 && candidate.local_trace > 0;
      }
      if (local_ok)
      {
        candidate.signed_trace_dofs.assign(
            static_cast<std::size_t>(candidate.mesh->GetNE()) *
                candidate.local_trace,
            0);
        for (int element = 0; element < candidate.mesh->GetNE(); ++element)
        {
          mfem::Array<int> faces, orientations;
          candidate.mesh->GetElementEdges(element, faces, orientations);
          int offset = 0;
          for (int local_face = 0; local_face < faces.Size(); ++local_face)
          {
            mfem::Array<int> vdofs;
            candidate.trace_space->GetFaceVDofs(faces[local_face], vdofs);
            if (vdofs.Size() <= 0 ||
                offset + vdofs.Size() > candidate.local_trace)
            {
              local_ok = false;
              break;
            }
            std::copy(vdofs.begin(), vdofs.end(),
                      candidate.signed_trace_dofs.begin() +
                          static_cast<std::size_t>(element) *
                              candidate.local_trace +
                          offset);
            offset += vdofs.Size();
          }
          local_ok = local_ok && offset == candidate.local_trace;
          if (!local_ok) { break; }
        }
      }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
    local_ok = trace::BuildDGInterfaceTraceExchangePlan(
        *candidate.trace_space, candidate.exchange_plan);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
    candidate.old_interior = std::make_unique<mfem::ParGridFunction>(
        candidate.interior_space.get());
    candidate.old_trace = std::make_unique<mfem::ParGridFunction>(
        candidate.trace_space.get());
    candidate.current_interior = std::make_unique<mfem::ParGridFunction>(
        candidate.interior_space.get());
    candidate.current_trace = std::make_unique<mfem::ParGridFunction>(
        candidate.trace_space.get());
    auto exact = [](const mfem::Vector& point, mfem::Vector& value)
    {
      ConservedState state{};
      if (!ManufacturedConservative(point[0], point[1], state).ok())
      {
        throw std::runtime_error("manufactured state projection failed");
      }
      value.SetSize(conserved_components);
      for (int component = 0; component < conserved_components; ++component)
      {
        value[component] = state[component];
      }
    };
    auto perturbed = [](const mfem::Vector& point, mfem::Vector& value)
    {
      PrimitiveState primitive{};
      if (!ManufacturedPrimitive(point[0], point[1], primitive).ok())
      {
        throw std::runtime_error("manufactured primitive failed");
      }
      const double psi_one = std::sin(2.0 * pi * point[0]) *
          std::cos(2.0 * pi * point[1]);
      const double psi_two = std::cos(2.0 * pi * point[0]) *
          std::sin(2.0 * pi * point[1]);
      const double psi_three =
          std::sin(2.0 * pi * (point[0] + point[1]));
      primitive[0] *= 1.0 + 0.04 * psi_one;
      primitive[1] += 0.03 * psi_two;
      primitive[2] -= 0.03 * psi_one;
      primitive[3] *= 1.0 + 0.04 * psi_three;
      ConservedState state{};
      if (!ConservativeFromPrimitive(primitive, state).ok())
      {
        throw std::runtime_error("perturbed state projection failed");
      }
      value.SetSize(conserved_components);
      for (int component = 0; component < conserved_components; ++component)
      {
        value[component] = state[component];
      }
    };
    mfem::VectorFunctionCoefficient exact_coefficient(
        conserved_components, exact);
    mfem::VectorFunctionCoefficient perturbed_coefficient(
        conserved_components, perturbed);
    candidate.old_interior->ProjectCoefficient(exact_coefficient);
    candidate.current_interior->ProjectCoefficient(perturbed_coefficient);
    local_ok = ProjectTrace(*candidate.mesh, *candidate.trace_space,
                            *candidate.old_trace, false) &&
        ProjectTrace(*candidate.mesh, *candidate.trace_space,
                     *candidate.current_trace, true);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  mfem::Vector owned_current_trace;
  mfem::Vector refreshed_current_trace;
  try
  {
    owned_current_trace.SetSize(candidate.trace_space->GetTrueVSize());
    refreshed_current_trace.SetSize(candidate.trace_space->GetVSize());
    candidate.current_trace->ParallelProject(owned_current_trace);
    refreshed_current_trace = std::numeric_limits<double>::quiet_NaN();
    local_ok = FiniteValues(VectorValues(owned_current_trace));
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
    local_ok = candidate.exchange_plan.Distribute(
        *candidate.trace_space, owned_current_trace,
        refreshed_current_trace);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  try
  {
    local_ok = AssignValues(*candidate.current_trace,
                            VectorValues(refreshed_current_trace));
    if (local_ok)
    {
      mfem::Vector owner_round_trip(candidate.trace_space->GetTrueVSize());
      candidate.current_trace->ParallelProject(owner_round_trip);
      local_ok = VectorValues(owner_round_trip) ==
          VectorValues(owned_current_trace);
    }
    if (local_ok)
    {
      double minimum_density = 0.0;
      double minimum_pressure = 0.0;
      local_ok = PrescreenState(
          *candidate.mesh, *candidate.interior_space,
          *candidate.trace_space, *candidate.current_interior,
          *candidate.current_trace, quadrature_order_, minimum_density,
          minimum_pressure);
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!implementation_->CollectiveAll(local_ok)) { return false; }

  std::swap(implementation_->mesh, candidate.mesh);
  std::swap(implementation_->interior_collection,
            candidate.interior_collection);
  std::swap(implementation_->trace_collection, candidate.trace_collection);
  std::swap(implementation_->interior_space, candidate.interior_space);
  std::swap(implementation_->trace_space, candidate.trace_space);
  std::swap(implementation_->old_interior, candidate.old_interior);
  std::swap(implementation_->old_trace, candidate.old_trace);
  std::swap(implementation_->current_interior, candidate.current_interior);
  std::swap(implementation_->current_trace, candidate.current_trace);
  std::swap(implementation_->exchange_plan, candidate.exchange_plan);
  implementation_->signed_trace_dofs.swap(candidate.signed_trace_dofs);
  implementation_->local_interior = candidate.local_interior;
  implementation_->local_trace = candidate.local_trace;
  implementation_->elements_per_direction = elements_per_direction;
  implementation_->published_status = SmoothEulerMPIStatus::Ready;
  ++implementation_->published_generation;
  implementation_->published_evidence = {};
  implementation_->last_failure_point_reached =
      SmoothEulerMPIFailurePoint::None;
  return true;
}

bool SmoothEulerMPIExperiment::Execute(
    const SmoothEulerMPIFailureInjection& injection)
{
  auto& state = *implementation_;
  const int local_point = static_cast<int>(injection.point);
  const int local_rank = injection.rank;
  int minimum_arguments[2] = {0, 0};
  int maximum_arguments[2] = {0, 0};
  const int local_arguments[2] = {local_point, local_rank};
  MPI_Allreduce(local_arguments, minimum_arguments, 2, MPI_INT, MPI_MIN,
                state.communicator);
  MPI_Allreduce(local_arguments, maximum_arguments, 2, MPI_INT, MPI_MAX,
                state.communicator);
  const bool valid_injection = local_point >=
          static_cast<int>(SmoothEulerMPIFailurePoint::None) &&
      local_point <= static_cast<int>(SmoothEulerMPIFailurePoint::Commit) &&
      ((injection.point == SmoothEulerMPIFailurePoint::None &&
        injection.rank == -1) ||
       (injection.point != SmoothEulerMPIFailurePoint::None &&
        injection.rank >= 0 && injection.rank < state.ranks));
  bool local_ok = valid_injection &&
      minimum_arguments[0] == maximum_arguments[0] &&
      minimum_arguments[1] == maximum_arguments[1] && state.mesh &&
      state.interior_space &&
      state.trace_space && state.old_interior && state.current_interior &&
      state.current_trace && state.exchange_plan.IsConsistent() &&
      state.published_status != SmoothEulerMPIStatus::Uninitialized &&
      state.elements_per_direction == 3 && state.local_interior > 0 &&
      state.local_trace > 0;
  if (!state.CollectiveAll(local_ok)) { return false; }
  state.last_failure_point_reached = SmoothEulerMPIFailurePoint::None;

  SmoothEulerMPIEvidence candidate_evidence;
  candidate_evidence.status = SmoothEulerMPIStatus::Success;
  candidate_evidence.rank = state.rank;
  candidate_evidence.ranks = state.ranks;
  candidate_evidence.polynomial_order = polynomial_order_;
  candidate_evidence.elements_per_direction = state.elements_per_direction;
  candidate_evidence.entry_generation = state.published_generation;
  try
  {
    candidate_evidence.entry_interior = VectorValues(*state.current_interior);
    candidate_evidence.entry_trace = VectorValues(*state.current_trace);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  if (!state.AdmitPhase(injection, SmoothEulerMPIFailurePoint::Assembly))
  {
    return false;
  }
  LocalSystem local_system;
  try
  {
    local_ok = local_ok && AssembleLocalSystem(
        *state.mesh, *state.interior_space, *state.trace_space,
        *state.old_interior, *state.current_interior, *state.current_trace,
        quadrature_order_, state.local_interior, state.local_trace,
        local_system);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  candidate_evidence.initial_nonlinear_residual =
      GlobalResidualNorm(state.communicator, *state.trace_space, local_system,
                         state.signed_trace_dofs);
  local_ok = IsFinite(candidate_evidence.initial_nonlinear_residual) &&
      candidate_evidence.initial_nonlinear_residual > 0.0;
  if (!state.CollectiveAll(local_ok)) { return false; }

  try
  {
    candidate_evidence.linearization.local_elements = local_system.elements;
    candidate_evidence.linearization.interior_dofs =
        local_system.interior_dofs;
    candidate_evidence.linearization.trace_dofs = local_system.trace_dofs;
    candidate_evidence.linearization.residual_interior =
        local_system.residual_interior;
    candidate_evidence.linearization.residual_trace =
        local_system.residual_trace;
    candidate_evidence.linearization.a = local_system.a;
    candidate_evidence.linearization.b = local_system.b;
    candidate_evidence.linearization.c = local_system.c;
    candidate_evidence.linearization.d = local_system.d;
    candidate_evidence.linearization.signed_trace_dofs =
        state.signed_trace_dofs;
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  std::vector<double> schur, reduced_rhs;
  core::HostReferenceEngine condensation;
  if (!state.AdmitPhase(injection, SmoothEulerMPIFailurePoint::Condensation))
  {
    return false;
  }
  try
  {
    const int batches = local_system.elements;
    const int ni = local_system.interior_dofs;
    const int nt = local_system.trace_dofs;
    schur.assign(static_cast<std::size_t>(batches) * nt * nt, 0.0);
    reduced_rhs.assign(static_cast<std::size_t>(batches) * nt, 0.0);
    const LocalBlockShape shape{ni / conserved_components,
                                nt / conserved_components,
                                conserved_components,
                                conserved_components};
    const core::LocalSystemBatchView input{
        shape, AsConst(View(local_system.a.data(), batches, ni, ni)),
        AsConst(View(local_system.b.data(), batches, ni, nt)),
        AsConst(View(local_system.c.data(), batches, nt, ni)),
        AsConst(View(local_system.d.data(), batches, nt, nt)),
        AsConst(View(local_system.f.data(), batches, ni, 1)),
        AsConst(View(local_system.g.data(), batches, nt, 1))};
    local_ok = condensation.Condense(
            input, {View(schur.data(), batches, nt, nt),
                    View(reduced_rhs.data(), batches, nt, 1)}) ==
            core::Status::success &&
        FiniteValues(schur) && FiniteValues(reduced_rhs);
    if (local_ok)
    {
      candidate_evidence.condensation.schur = schur;
      candidate_evidence.condensation.right_hand_side = reduced_rhs;
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  std::unique_ptr<mfem::HypreParMatrix> true_matrix;
  std::unique_ptr<mfem::SparseMatrix> local_matrix;
  mfem::Vector local_trace_rhs;
  mfem::Vector true_rhs_values;
  if (!state.AdmitPhase(injection, SmoothEulerMPIFailurePoint::TrueAssembly))
  {
    return false;
  }
  try
  {
    local_trace_rhs.SetSize(state.trace_space->GetVSize());
    true_rhs_values.SetSize(state.trace_space->GetTrueVSize());
    local_matrix =
        std::make_unique<mfem::SparseMatrix>(state.trace_space->GetVSize());
    local_trace_rhs = 0.0;
    for (int element = 0; element < local_system.elements; ++element)
    {
      mfem::Array<int> dofs(local_system.trace_dofs);
      mfem::DenseMatrix local_schur(local_system.trace_dofs);
      for (int row = 0; row < local_system.trace_dofs; ++row)
      {
        const int signed_dof = state.signed_trace_dofs[
            static_cast<std::size_t>(element) * local_system.trace_dofs + row];
        dofs[row] = signed_dof;
        double sign = 1.0;
        const int local_dof = DecodeSignedVDof(signed_dof, sign);
        if (local_dof < 0 || local_dof >= local_trace_rhs.Size())
        {
          local_ok = false;
          break;
        }
        local_trace_rhs[local_dof] += sign * reduced_rhs[
            static_cast<std::size_t>(element) * local_system.trace_dofs + row];
        for (int column = 0; column < local_system.trace_dofs; ++column)
        {
          local_schur(row, column) = schur[
              static_cast<std::size_t>(element) * local_system.trace_dofs *
                  local_system.trace_dofs +
              static_cast<std::size_t>(row) * local_system.trace_dofs +
              column];
        }
      }
      if (!local_ok) { break; }
      local_matrix->AddSubMatrix(dofs, dofs, local_schur);
    }
    if (local_ok)
    {
      local_matrix->Finalize();
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  std::unique_ptr<mfem::ParBilinearForm> parallel_form;
  try
  {
    parallel_form = std::make_unique<mfem::ParBilinearForm>(
        state.trace_space.get());
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  try
  {
    true_matrix.reset(parallel_form->ParallelAssemble(local_matrix.get()));
    local_ok = true_matrix &&
        true_matrix->Height() == state.trace_space->GetTrueVSize() &&
        true_matrix->GetGlobalNumRows() ==
            state.trace_space->GlobalTrueVSize() &&
        true_matrix->GetGlobalNumCols() ==
            state.trace_space->GlobalTrueVSize() &&
        true_matrix->GetGlobalNumRows() <=
            std::numeric_limits<int>::max();
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  try
  {
    state.trace_space->Dof_TrueDof_Matrix()->MultTranspose(
        local_trace_rhs, true_rhs_values);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  try
  {
    candidate_evidence.condensation.local_assembled_right_hand_side =
        VectorValues(local_trace_rhs);
    local_ok = FiniteValues(VectorValues(true_rhs_values));
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  std::unique_ptr<mfem::HypreParVector> true_rhs;
  std::unique_ptr<mfem::HypreParVector> true_solution;
  std::unique_ptr<mfem::HypreParVector> true_probe;
  std::unique_ptr<mfem::HypreParVector> true_probe_action;
  if (!state.AdmitPhase(injection, SmoothEulerMPIFailurePoint::TraceSolve))
  {
    return false;
  }
  std::unique_ptr<mfem::HypreILU> ilu;
  std::unique_ptr<mfem::HypreGMRES> gmres;
  try
  {
    true_rhs = std::make_unique<mfem::HypreParVector>(
        state.trace_space.get());
    true_solution = std::make_unique<mfem::HypreParVector>(
        state.trace_space.get());
    true_probe = std::make_unique<mfem::HypreParVector>(
        state.trace_space.get());
    true_probe_action = std::make_unique<mfem::HypreParVector>(
        state.trace_space.get());
    local_ok = AssignValues(*true_rhs, VectorValues(true_rhs_values));
    if (local_ok)
    {
      *true_solution = 0.0;
      ilu = std::make_unique<mfem::HypreILU>();
      ilu->SetType(0);
      ilu->SetLevelOfFill(0);
      ilu->SetMaxIter(1);
      ilu->SetTol(0.0);
      ilu->SetLocalReordering(0);
      ilu->SetPrintLevel(0);
      gmres = std::make_unique<mfem::HypreGMRES>(state.communicator);
      gmres->SetTol(1.0e-12);
      gmres->SetAbsTol(0.0);
      const int global_trace_dofs = static_cast<int>(
          true_matrix->GetGlobalNumRows());
      gmres->SetMaxIter(std::max(100, global_trace_dofs));
      gmres->SetKDim(std::min(100, std::max(20, global_trace_dofs)));
      gmres->SetLogging(1);
      gmres->SetPrintLevel(0);
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  try
  {
    gmres->SetPreconditioner(*ilu);
    gmres->SetOperator(*true_matrix);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  try
  {
    gmres->Mult(*true_rhs, *true_solution);
    gmres->GetNumIterations(candidate_evidence.gmres_iterations);
    gmres->GetFinalResidualNorm(
        candidate_evidence.gmres_reported_relative_residual);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  std::unique_ptr<mfem::HypreParVector> residual;
  try
  {
    residual = std::make_unique<mfem::HypreParVector>(
        state.trace_space.get());
    local_ok = candidate_evidence.gmres_iterations > 0 &&
        IsFinite(candidate_evidence.gmres_reported_relative_residual) &&
        FiniteValues(VectorValues(*true_solution));
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  try
  {
    true_matrix->Mult(*true_solution, *residual);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  try
  {
    *residual -= *true_rhs;
    local_ok = FiniteValues(VectorValues(*residual));
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  const double residual_squared = mfem::InnerProduct(
      state.communicator, *residual, *residual);
  const double rhs_squared = mfem::InnerProduct(
      state.communicator, *true_rhs, *true_rhs);
  candidate_evidence.true_relative_residual = std::sqrt(residual_squared) /
      std::max(std::sqrt(rhs_squared), 1.0e-300);
  local_ok = IsFinite(candidate_evidence.true_relative_residual) &&
      candidate_evidence.true_relative_residual <= 1.0e-10;
  if (!state.CollectiveAll(local_ok)) { return false; }

  try
  {
    *true_probe = std::numeric_limits<double>::quiet_NaN();
    double* const probe_data = true_probe->HostWrite();
    local_ok = probe_data != nullptr;
    for (const trace::TraceMapEntry& entry :
         state.exchange_plan.map.dofs.entries)
      if (entry.ownership == Ownership::owned && local_ok)
      {
        probe_data[entry.local_true_dof] =
            0.25 + std::sin(0.01 *
                static_cast<double>(entry.global_true_dof + 1));
      }
    local_ok = local_ok && FiniteValues(VectorValues(*true_probe));
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  try
  {
    true_matrix->Mult(*true_probe, *true_probe_action);
    local_ok = FiniteValues(VectorValues(*true_probe_action));
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  std::unique_ptr<mfem::HypreParVector> true_basis;
  std::unique_ptr<mfem::HypreParVector> true_basis_action;
  std::vector<double> true_basis_actions;
  const int global_trace_dofs =
      static_cast<int>(true_matrix->GetGlobalNumRows());
  try
  {
    true_basis = std::make_unique<mfem::HypreParVector>(
        state.trace_space.get());
    true_basis_action = std::make_unique<mfem::HypreParVector>(
        state.trace_space.get());
    true_basis_actions.assign(
        static_cast<std::size_t>(global_trace_dofs) *
            state.trace_space->GetTrueVSize(),
        0.0);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  for (int column = 0; column < global_trace_dofs; ++column)
  {
    try
    {
      *true_basis = 0.0;
      for (const trace::TraceMapEntry& entry :
           state.exchange_plan.map.dofs.entries)
        if (entry.ownership == Ownership::owned &&
            entry.global_true_dof == column)
        {
          (*true_basis)[entry.local_true_dof] = 1.0;
        }
    }
    catch (const std::exception&)
    {
      local_ok = false;
    }
    if (!state.CollectiveAll(local_ok)) { return false; }
    try
    {
      true_matrix->Mult(*true_basis, *true_basis_action);
      const double* const action = true_basis_action->HostRead();
      local_ok = action != nullptr;
      if (local_ok)
      {
        for (int row = 0; row < true_basis_action->Size(); ++row)
        {
          local_ok = local_ok && IsFinite(action[row]);
        }
      }
      if (local_ok)
      {
        std::copy(action, action + true_basis_action->Size(),
                  true_basis_actions.begin() +
                      static_cast<std::size_t>(column) *
                          true_basis_action->Size());
      }
    }
    catch (const std::exception&)
    {
      local_ok = false;
    }
    if (!state.CollectiveAll(local_ok)) { return false; }
  }
  try
  {
    candidate_evidence.true_right_hand_side = VectorValues(*true_rhs);
    candidate_evidence.true_trace_increment = VectorValues(*true_solution);
    candidate_evidence.true_probe = VectorValues(*true_probe);
    candidate_evidence.true_probe_action = VectorValues(*true_probe_action);
    candidate_evidence.true_basis_actions = std::move(true_basis_actions);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  mfem::Vector distributed_increment;
  try
  {
    distributed_increment.SetSize(state.trace_space->GetVSize());
    distributed_increment = 0.0;
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  if (!state.AdmitPhase(injection, SmoothEulerMPIFailurePoint::Distribution))
  {
    return false;
  }
  try
  {
    local_ok = state.exchange_plan.Distribute(
        *state.trace_space, *true_solution, distributed_increment);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  try
  {
    candidate_evidence.distributed_trace_increment =
        VectorValues(distributed_increment);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  std::vector<double> local_trace_increment;
  std::vector<double> recovered;
  if (!state.AdmitPhase(injection, SmoothEulerMPIFailurePoint::Recovery))
  {
    return false;
  }
  try
  {
    local_trace_increment.assign(
        static_cast<std::size_t>(local_system.elements) *
            local_system.trace_dofs,
        0.0);
    recovered.assign(
        static_cast<std::size_t>(local_system.elements) *
            local_system.interior_dofs,
        0.0);
    const double* const local_values = distributed_increment.HostRead();
    local_ok = local_values != nullptr;
    if (local_ok)
    {
      for (int element = 0; element < local_system.elements; ++element)
        for (int local = 0; local < local_system.trace_dofs; ++local)
        {
          double sign = 1.0;
          const int vdof = DecodeSignedVDof(state.signed_trace_dofs[
              static_cast<std::size_t>(element) * local_system.trace_dofs +
              local], sign);
          if (vdof < 0 || vdof >= distributed_increment.Size())
          {
            local_ok = false;
            break;
          }
          local_trace_increment[
              static_cast<std::size_t>(element) * local_system.trace_dofs +
              local] = sign * local_values[vdof];
        }
    }
    local_ok = local_ok &&
        condensation.Recover(
            AsConst(View(local_trace_increment.data(), local_system.elements,
                         local_system.trace_dofs, 1)),
            View(recovered.data(), local_system.elements,
                 local_system.interior_dofs, 1)) == core::Status::success &&
        FiniteValues(recovered);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  try
  {
    candidate_evidence.recovered_interior_increment = recovered;
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  std::vector<double> grid_interior_increment;
  std::vector<int> interior_census;
  try
  {
    grid_interior_increment.assign(
        static_cast<std::size_t>(state.current_interior->Size()), 0.0);
    interior_census.assign(grid_interior_increment.size(), 0);
    for (int element = 0; element < local_system.elements; ++element)
    {
      mfem::Array<int> interior_vdofs;
      state.interior_space->GetElementVDofs(element, interior_vdofs);
      if (interior_vdofs.Size() != local_system.interior_dofs)
      {
        local_ok = false;
        break;
      }
      for (int local = 0; local < local_system.interior_dofs; ++local)
      {
        double sign = 1.0;
        const int vdof = DecodeSignedVDof(interior_vdofs[local], sign);
        if (vdof < 0 || vdof >= state.current_interior->Size())
        {
          local_ok = false;
          break;
        }
        grid_interior_increment[static_cast<std::size_t>(vdof)] = sign *
            recovered[static_cast<std::size_t>(element) *
                          local_system.interior_dofs +
                      local];
        ++interior_census[static_cast<std::size_t>(vdof)];
      }
    }
    local_ok = local_ok &&
        std::all_of(interior_census.begin(), interior_census.end(),
                    [](const int count) { return count == 1; }) &&
        FiniteValues(grid_interior_increment);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  std::unique_ptr<mfem::ParGridFunction> candidate_interior;
  std::unique_ptr<mfem::ParGridFunction> candidate_trace;
  bool accepted = false;
  double accepted_density = 0.0;
  double accepted_pressure = 0.0;
  double accepted_residual = 0.0;
  double accepted_damping = 0.0;
  if (!state.AdmitPhase(
          injection, SmoothEulerMPIFailurePoint::CandidateValidation))
  {
    return false;
  }
  for (int trial = 0; trial < maximum_line_search_trials; ++trial)
  {
    const double damping = std::ldexp(1.0, -trial);
    local_ok = true;
    bool local_admissible = false;
    LocalSystem candidate_system;
    double local_density = 0.0;
    double local_pressure = 0.0;
    try
    {
      candidate_interior = std::make_unique<mfem::ParGridFunction>(
          state.interior_space.get());
      candidate_trace = std::make_unique<mfem::ParGridFunction>(
          state.trace_space.get());
      const double* const entry_interior = state.current_interior->HostRead();
      const double* const entry_trace = state.current_trace->HostRead();
      double* const next_interior = candidate_interior->HostWrite();
      double* const next_trace = candidate_trace->HostWrite();
      local_ok = local_ok && entry_interior && entry_trace && next_interior &&
          next_trace &&
          recovered.size() ==
              static_cast<std::size_t>(state.current_interior->Size()) &&
          distributed_increment.Size() == state.current_trace->Size();
      if (local_ok)
      {
        const double* const trace_increment =
            distributed_increment.HostRead();
        local_ok = trace_increment != nullptr;
        for (int index = 0;
             index < candidate_interior->Size() && local_ok; ++index)
        {
          next_interior[index] = entry_interior[index] +
              damping *
                  grid_interior_increment[static_cast<std::size_t>(index)];
        }
        for (int index = 0;
             index < candidate_trace->Size() && local_ok; ++index)
        {
          next_trace[index] = entry_trace[index] +
              damping * trace_increment[index];
        }
        if (local_ok)
        {
          local_admissible = PrescreenState(
              *state.mesh, *state.interior_space, *state.trace_space,
              *candidate_interior, *candidate_trace, quadrature_order_,
              local_density, local_pressure);
          if (local_admissible)
          {
            local_admissible = AssembleLocalSystem(
                *state.mesh, *state.interior_space, *state.trace_space,
                *state.old_interior, *candidate_interior, *candidate_trace,
                quadrature_order_, state.local_interior, state.local_trace,
                candidate_system);
          }
        }
      }
    }
    catch (const std::exception&)
    {
      local_ok = false;
    }
    if (!state.CollectiveAll(local_ok)) { return false; }
    if (!state.CollectiveAll(local_admissible)) { continue; }
    const double candidate_residual =
        GlobalResidualNorm(state.communicator, *state.trace_space,
                           candidate_system, state.signed_trace_dofs);
    double global_density = 0.0;
    double global_pressure = 0.0;
    MPI_Allreduce(&local_density, &global_density, 1, MPI_DOUBLE, MPI_MIN,
                  state.communicator);
    MPI_Allreduce(&local_pressure, &global_pressure, 1, MPI_DOUBLE, MPI_MIN,
                  state.communicator);
    const bool local_accepted = IsFinite(candidate_residual) &&
        candidate_residual <=
            (1.0 - armijo_constant * damping) *
                candidate_evidence.initial_nonlinear_residual &&
        IsFinite(global_density) && global_density > 0.0 &&
        IsFinite(global_pressure) && global_pressure > 0.0;
    accepted = state.CollectiveAll(local_accepted);
    if (accepted)
    {
      accepted_density = global_density;
      accepted_pressure = global_pressure;
      accepted_residual = candidate_residual;
      accepted_damping = damping;
      break;
    }
  }
  if (!accepted) { return false; }

  if (!state.AdmitPhase(injection, SmoothEulerMPIFailurePoint::Commit))
  {
    return false;
  }
  local_ok = true;
  try
  {
    candidate_evidence.final_interior = VectorValues(*candidate_interior);
    candidate_evidence.final_trace = VectorValues(*candidate_trace);
    candidate_evidence.damping = accepted_damping;
    candidate_evidence.candidate_nonlinear_residual = accepted_residual;
    candidate_evidence.candidate_minimum_density = accepted_density;
    candidate_evidence.candidate_minimum_pressure = accepted_pressure;
    candidate_evidence.local_elements = state.mesh->GetNE();
    candidate_evidence.local_trace_dofs = state.trace_space->GetVSize();
    candidate_evidence.local_true_trace_dofs =
        state.trace_space->GetTrueVSize();
    candidate_evidence.global_true_trace_dofs =
        state.exchange_plan.map.dofs.global_true_dof_count;
    candidate_evidence.owned_trace_dofs =
        state.exchange_plan.map.OwnedCount();
    candidate_evidence.ghost_trace_dofs =
        state.exchange_plan.map.GhostCount();
    candidate_evidence.shared_faces = state.mesh->GetNSharedFaces();
    candidate_evidence.negative_signed_trace_dofs =
        static_cast<int>(std::count_if(
            state.signed_trace_dofs.begin(), state.signed_trace_dofs.end(),
            [](const int dof) { return dof < 0; }));
    candidate_evidence.committed_generation = state.published_generation + 1;
    candidate_evidence.passed = true;
    local_ok = local_ok && FiniteValues(candidate_evidence.final_interior) &&
        FiniteValues(candidate_evidence.final_trace);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  const unsigned long long local_ghosts =
      static_cast<unsigned long long>(candidate_evidence.ghost_trace_dofs);
  unsigned long long global_ghosts = 0;
  MPI_Allreduce(&candidate_evidence.local_elements,
                &candidate_evidence.global_elements, 1, MPI_INT, MPI_SUM,
                state.communicator);
  MPI_Allreduce(&local_ghosts, &global_ghosts, 1,
                MPI_UNSIGNED_LONG_LONG, MPI_SUM, state.communicator);
  candidate_evidence.global_ghost_trace_dofs =
      static_cast<std::size_t>(global_ghosts);
  MPI_Allreduce(&candidate_evidence.shared_faces,
                &candidate_evidence.global_shared_faces, 1, MPI_INT,
                MPI_SUM, state.communicator);
  MPI_Allreduce(&candidate_evidence.negative_signed_trace_dofs,
                &candidate_evidence.global_negative_signed_trace_dofs, 1,
                MPI_INT, MPI_SUM, state.communicator);

  state.current_interior = std::move(candidate_interior);
  state.current_trace = std::move(candidate_trace);
  state.published_generation = candidate_evidence.committed_generation;
  state.published_status = SmoothEulerMPIStatus::Success;
  state.published_evidence = std::move(candidate_evidence);
  return true;
}

SmoothEulerMPIStatus SmoothEulerMPIExperiment::status() const noexcept
{
  return implementation_ ? implementation_->published_status :
                           SmoothEulerMPIStatus::Uninitialized;
}

SmoothEulerMPIFailurePoint
SmoothEulerMPIExperiment::last_failure_point_reached() const noexcept
{
  return implementation_ ? implementation_->last_failure_point_reached :
                           SmoothEulerMPIFailurePoint::None;
}

std::uint64_t SmoothEulerMPIExperiment::generation() const noexcept
{
  return implementation_ ? implementation_->published_generation : 0;
}

int SmoothEulerMPIExperiment::elements_per_direction() const noexcept
{
  return implementation_ ? implementation_->elements_per_direction : 0;
}

const SmoothEulerMPIEvidence& SmoothEulerMPIExperiment::evidence() const noexcept
{
  static const SmoothEulerMPIEvidence empty{};
  return implementation_ ? implementation_->published_evidence : empty;
}

const mfem::ParMesh* SmoothEulerMPIExperiment::mesh() const noexcept
{
  return implementation_ ? implementation_->mesh.get() : nullptr;
}

const mfem::ParFiniteElementSpace*
SmoothEulerMPIExperiment::interior_space() const noexcept
{
  return implementation_ ? implementation_->interior_space.get() : nullptr;
}

const mfem::ParFiniteElementSpace*
SmoothEulerMPIExperiment::trace_space() const noexcept
{
  return implementation_ ? implementation_->trace_space.get() : nullptr;
}

const mfem::ParGridFunction*
SmoothEulerMPIExperiment::old_interior() const noexcept
{
  return implementation_ ? implementation_->old_interior.get() : nullptr;
}

const mfem::ParGridFunction*
SmoothEulerMPIExperiment::current_interior() const noexcept
{
  return implementation_ ? implementation_->current_interior.get() : nullptr;
}

const mfem::ParGridFunction*
SmoothEulerMPIExperiment::current_trace() const noexcept
{
  return implementation_ ? implementation_->current_trace.get() : nullptr;
}

const trace::DGInterfaceTraceMap*
SmoothEulerMPIExperiment::trace_map() const noexcept
{
  return implementation_ ? &implementation_->exchange_plan.map : nullptr;
}

} // namespace dg::hdg::euler

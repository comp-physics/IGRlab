#include "numerics/hdg/euler/smooth_euler_device.hpp"

#include "numerics/hdg/euler/smooth_euler.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dg::hdg::euler
{
namespace
{
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double physics_gate = 2.0e-6;
constexpr double constant_gate = 5.0e-12;
constexpr double local_gate = 5.0e-11;
constexpr double solve_gate = 5.0e-9;
constexpr double true_residual_gate = 1.0e-10;

template <typename T>
StridedBlockView<T> View(T* data, const LocalIndex batches,
                         const LocalIndex rows, const LocalIndex columns) noexcept
{
  return {data, batches, rows, columns, rows * columns, columns, 1};
}

template <typename T>
StridedBlockView<const T> ConstView(T* data, const LocalIndex batches,
                                    const LocalIndex rows,
                                    const LocalIndex columns) noexcept
{
  return AsConst(View(data, batches, rows, columns));
}

MFEM_HOST_DEVICE int LocalByNodes(const int scalar, const int scalar_count,
                                  const int component)
{
  return scalar + scalar_count * component;
}

MFEM_HOST_DEVICE bool FiniteState(const double* state)
{
  for (int component = 0; component < conserved_components; ++component)
  {
    if (!isfinite(state[component])) { return false; }
  }
  return true;
}

MFEM_HOST_DEVICE bool FluxAndJacobian(const double* state, const double nx,
                                      const double ny, double* flux,
                                      double* jacobian)
{
  if (!FiniteState(state) || !isfinite(nx) || !isfinite(ny) ||
      (nx == 0.0 && ny == 0.0) || !(state[0] > 0.0))
  {
    return false;
  }
  const double rho = state[0];
  const double mx = state[1];
  const double my = state[2];
  const double energy = state[3];
  const double kinetic = 0.5 * (mx * mx + my * my) / rho;
  const double pressure = (gamma - 1.0) * (energy - kinetic);
  if (!(pressure > 0.0) || !isfinite(pressure)) { return false; }
  const double q = mx * nx + my * ny;
  flux[0] = q;
  flux[1] = mx * q / rho + pressure * nx;
  flux[2] = my * q / rho + pressure * ny;
  flux[3] = (energy + pressure) * q / rho;

  const double pressure_rho =
      (gamma - 1.0) * 0.5 * (mx * mx + my * my) / (rho * rho);
  const double pressure_mx = -(gamma - 1.0) * mx / rho;
  const double pressure_my = -(gamma - 1.0) * my / rho;
  const double pressure_energy = gamma - 1.0;
  const double enthalpy = energy + pressure;
  for (int entry = 0; entry < conserved_components * conserved_components;
       ++entry)
  {
    jacobian[entry] = 0.0;
  }
  jacobian[0 * conserved_components + 1] = nx;
  jacobian[0 * conserved_components + 2] = ny;
  jacobian[1 * conserved_components + 0] =
      -mx * q / (rho * rho) + pressure_rho * nx;
  jacobian[1 * conserved_components + 1] =
      q / rho + mx * nx / rho + pressure_mx * nx;
  jacobian[1 * conserved_components + 2] =
      mx * ny / rho + pressure_my * nx;
  jacobian[1 * conserved_components + 3] = pressure_energy * nx;
  jacobian[2 * conserved_components + 0] =
      -my * q / (rho * rho) + pressure_rho * ny;
  jacobian[2 * conserved_components + 1] =
      my * nx / rho + pressure_mx * ny;
  jacobian[2 * conserved_components + 2] =
      q / rho + my * ny / rho + pressure_my * ny;
  jacobian[2 * conserved_components + 3] = pressure_energy * ny;
  jacobian[3 * conserved_components + 0] =
      pressure_rho * q / rho - enthalpy * q / (rho * rho);
  jacobian[3 * conserved_components + 1] =
      pressure_mx * q / rho + enthalpy * nx / rho;
  jacobian[3 * conserved_components + 2] =
      pressure_my * q / rho + enthalpy * ny / rho;
  jacobian[3 * conserved_components + 3] =
      (1.0 + pressure_energy) * q / rho;
  return FiniteState(flux) && FiniteState(jacobian) &&
         FiniteState(jacobian + conserved_components) &&
         FiniteState(jacobian + 2 * conserved_components) &&
         FiniteState(jacobian + 3 * conserved_components);
}

MFEM_HOST_DEVICE void ManufacturedState(const double x, const double y,
                                        double* state)
{
  const double sx = sin(2.0 * pi * x);
  const double sy = sin(2.0 * pi * y);
  const double cx = cos(2.0 * pi * x);
  const double cy = cos(2.0 * pi * y);
  const double rho = 1.0 + 0.1 * sx * sy;
  const double u = 0.35 + 0.05 * cx * sy;
  const double v = -0.20 + 0.05 * sx * cy;
  const double pressure = 1.0 + 0.1 * cx * cy;
  state[0] = rho;
  state[1] = rho * u;
  state[2] = rho * v;
  state[3] = pressure / (gamma - 1.0) +
             0.5 * rho * (u * u + v * v);
}

MFEM_HOST_DEVICE bool ManufacturedSourceDevice(const double x, const double y,
                                                double* source)
{
  const double sx = sin(2.0 * pi * x);
  const double sy = sin(2.0 * pi * y);
  const double cx = cos(2.0 * pi * x);
  const double cy = cos(2.0 * pi * y);
  double state[conserved_components];
  ManufacturedState(x, y, state);
  const double rho = state[0];
  const double u = state[1] / rho;
  const double v = state[2] / rho;
  const double derivative_rho_x = 0.2 * pi * cx * sy;
  const double derivative_rho_y = 0.2 * pi * sx * cy;
  const double derivative_u_x = -0.1 * pi * sx * sy;
  const double derivative_u_y = 0.1 * pi * cx * cy;
  const double derivative_v_x = 0.1 * pi * cx * cy;
  const double derivative_v_y = -0.1 * pi * sx * sy;
  const double derivative_p_x = -0.2 * pi * sx * cy;
  const double derivative_p_y = -0.2 * pi * cx * sy;
  double derivative_x[conserved_components];
  double derivative_y[conserved_components];
  derivative_x[0] = derivative_rho_x;
  derivative_x[1] = derivative_rho_x * u + rho * derivative_u_x;
  derivative_x[2] = derivative_rho_x * v + rho * derivative_v_x;
  derivative_x[3] = derivative_p_x / (gamma - 1.0) +
      0.5 * derivative_rho_x * (u * u + v * v) +
      rho * (u * derivative_u_x + v * derivative_v_x);
  derivative_y[0] = derivative_rho_y;
  derivative_y[1] = derivative_rho_y * u + rho * derivative_u_y;
  derivative_y[2] = derivative_rho_y * v + rho * derivative_v_y;
  derivative_y[3] = derivative_p_y / (gamma - 1.0) +
      0.5 * derivative_rho_y * (u * u + v * v) +
      rho * (u * derivative_u_y + v * derivative_v_y);
  double unused_flux[conserved_components];
  double jacobian_x[conserved_components * conserved_components];
  double jacobian_y[conserved_components * conserved_components];
  if (!FluxAndJacobian(state, 1.0, 0.0, unused_flux, jacobian_x) ||
      !FluxAndJacobian(state, 0.0, 1.0, unused_flux, jacobian_y))
  {
    return false;
  }
  for (int row = 0; row < conserved_components; ++row)
  {
    source[row] = 0.0;
    for (int column = 0; column < conserved_components; ++column)
    {
      source[row] += jacobian_x[row * conserved_components + column] *
                         derivative_x[column] +
                     jacobian_y[row * conserved_components + column] *
                         derivative_y[column];
    }
  }
  return FiniteState(source);
}

bool AssignVector(mfem::Vector& destination, const std::vector<double>& source)
{
  if (source.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    return false;
  }
  destination.SetSize(static_cast<int>(source.size()));
  destination.UseDevice(true);
  if (!source.empty())
  {
    std::copy(source.begin(), source.end(), destination.HostWrite());
  }
  return true;
}

bool SizeAndZero(mfem::Vector& vector, const std::size_t size)
{
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    return false;
  }
  vector.SetSize(static_cast<int>(size));
  vector.UseDevice(true);
  std::fill_n(vector.HostWrite(), size, 0.0);
  return true;
}

bool VectorFinite(const mfem::Vector& vector)
{
  const double* values = vector.HostRead();
  for (int entry = 0; entry < vector.Size(); ++entry)
  {
    if (!std::isfinite(values[entry])) { return false; }
  }
  return true;
}

struct CaseGraph
{
  std::unique_ptr<mfem::Mesh> mesh;
  std::unique_ptr<mfem::L2_FECollection> interior_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::FiniteElementSpace> interior_space;
  std::unique_ptr<mfem::FiniteElementSpace> trace_space;
  std::unique_ptr<mfem::GridFunction> old_interior;
  std::unique_ptr<mfem::GridFunction> current_interior;
  std::unique_ptr<mfem::GridFunction> current_trace;
  std::vector<int> signed_trace_dofs;
  int elements = 0;
  int scalar_dofs = 0;
  int face_scalar_dofs = 0;
  int local_interior = 0;
  int local_trace = 0;
  int volume_points = 0;
  int face_points = 0;
};

struct CaseStorage
{
  mfem::Vector old_coefficients, current_coefficients, trace_coefficients;
  mfem::Vector volume_shape, volume_derivative, volume_weight, volume_position;
  mfem::Vector face_interior_shape, face_trace_shape, face_weight, face_normal;
  mfem::Vector residual_r, residual_g, a, b, c, d, f, g, valid;
  mfem::Vector candidate_residual_r, candidate_residual_g;
  mfem::Vector candidate_a, candidate_b, candidate_c, candidate_d;
  mfem::Vector candidate_f, candidate_g, candidate_valid;
  std::uint64_t generation = 0;
};

struct ReferenceCase
{
  LocalLinearizationSnapshot local;
  int global_trace_dofs = 0;
  std::vector<double> trace_matrix;
  std::vector<double> trace_rhs;
  std::vector<double> trace_solution;
};

bool ProjectState(CaseGraph& graph, const bool manufactured)
{
  const ConservedState constant_state{{1.2, 0.48, -0.3, 2.3835}};
  auto coefficient = [manufactured, constant_state](const mfem::Vector& point,
                                                      mfem::Vector& value)
  {
    ConservedState state = constant_state;
    if (manufactured && !ManufacturedConservative(point[0], point[1], state).ok())
    {
      throw std::runtime_error("manufactured projection failed");
    }
    value.SetSize(conserved_components);
    for (int component = 0; component < conserved_components; ++component)
    {
      value[component] = state[component];
    }
  };
  try
  {
    mfem::VectorFunctionCoefficient exact(conserved_components, coefficient);
    graph.old_interior =
        std::make_unique<mfem::GridFunction>(graph.interior_space.get());
    graph.current_interior =
        std::make_unique<mfem::GridFunction>(graph.interior_space.get());
    graph.current_trace =
        std::make_unique<mfem::GridFunction>(graph.trace_space.get());
    graph.old_interior->ProjectCoefficient(exact);
    *graph.current_interior = *graph.old_interior;
    double* trace_values = graph.current_trace->HostWrite();
    for (int face = 0; face < graph.mesh->GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* face_fe = graph.trace_space->GetFaceElement(face);
      mfem::ElementTransformation* transformation =
          graph.mesh->GetFaceTransformation(face);
      if (face_fe == nullptr || transformation == nullptr ||
          face_fe->GetDof() != graph.face_scalar_dofs)
      {
        return false;
      }
      mfem::Array<int> vdofs;
      graph.trace_space->GetFaceVDofs(face, vdofs);
      if (vdofs.Size() != conserved_components * graph.face_scalar_dofs)
      {
        return false;
      }
      const mfem::IntegrationRule& nodes = face_fe->GetNodes();
      for (int scalar = 0; scalar < graph.face_scalar_dofs; ++scalar)
      {
        mfem::Vector point(dimension), value(conserved_components);
        transformation->Transform(nodes.IntPoint(scalar), point);
        coefficient(point, value);
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = ByNodesIndex(scalar, graph.face_scalar_dofs, component);
          double sign = 1.0;
          const int vdof = DecodeSignedVDof(vdofs[local], sign);
          if (vdof < 0 || vdof >= graph.current_trace->Size()) { return false; }
          trace_values[vdof] = sign * value[component];
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

bool BuildCase(const int p, const int N, const bool manufactured,
               CaseGraph& graph, CaseStorage& storage)
{
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        N, N, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(dimension));
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    graph.mesh = std::make_unique<mfem::Mesh>(mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations)));
    graph.interior_collection = std::make_unique<mfem::L2_FECollection>(
        p, dimension, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
    graph.trace_collection =
        std::make_unique<mfem::DG_Interface_FECollection>(
            p, dimension, mfem::FiniteElement::VALUE,
            mfem::BasisType::GaussLegendre);
    graph.interior_space = std::make_unique<mfem::FiniteElementSpace>(
        graph.mesh.get(), graph.interior_collection.get(), conserved_components,
        mfem::Ordering::byNODES);
    graph.trace_space = std::make_unique<mfem::FiniteElementSpace>(
        graph.mesh.get(), graph.trace_collection.get(), conserved_components,
        mfem::Ordering::byNODES);
    graph.elements = graph.mesh->GetNE();
    const mfem::FiniteElement* first_fe = graph.interior_space->GetFE(0);
    if (graph.elements != N * N || first_fe == nullptr) { return false; }
    graph.scalar_dofs = first_fe->GetDof();
    graph.local_interior = conserved_components * graph.scalar_dofs;
    mfem::Array<int> first_faces, first_orientations;
    graph.mesh->GetElementEdges(0, first_faces, first_orientations);
    if (first_faces.Size() != 4) { return false; }
    const mfem::FiniteElement* first_face_fe =
        graph.trace_space->GetFaceElement(first_faces[0]);
    if (first_face_fe == nullptr) { return false; }
    graph.face_scalar_dofs = first_face_fe->GetDof();
    graph.local_trace = first_faces.Size() * conserved_components *
                        graph.face_scalar_dofs;
    const int quadrature_order = 3 * p + 8;
    graph.volume_points =
        mfem::IntRules.Get(first_fe->GetGeomType(), quadrature_order).GetNPoints();
    graph.face_points = mfem::IntRules.Get(first_face_fe->GetGeomType(),
                                           quadrature_order).GetNPoints();
    if (graph.scalar_dofs <= 0 || graph.face_scalar_dofs <= 0 ||
        graph.local_interior <= 0 || graph.local_trace <= 0 ||
        graph.volume_points <= 0 || graph.face_points <= 0 ||
        !ProjectState(graph, manufactured))
    {
      return false;
    }

    std::vector<double> old_coefficients(
        static_cast<std::size_t>(graph.elements) * graph.local_interior);
    std::vector<double> current_coefficients(old_coefficients.size());
    std::vector<double> trace_coefficients(
        static_cast<std::size_t>(graph.elements) * graph.local_trace);
    graph.signed_trace_dofs.assign(trace_coefficients.size(), 0);
    std::vector<double> volume_shape(
        static_cast<std::size_t>(graph.elements) * graph.volume_points *
        graph.scalar_dofs);
    std::vector<double> volume_derivative(volume_shape.size() * dimension);
    std::vector<double> volume_weight(
        static_cast<std::size_t>(graph.elements) * graph.volume_points);
    std::vector<double> volume_position(volume_weight.size() * dimension);
    std::vector<double> face_interior_shape(
        static_cast<std::size_t>(graph.elements) * 4 * graph.face_points *
        graph.scalar_dofs);
    std::vector<double> face_trace_shape(
        static_cast<std::size_t>(graph.elements) * 4 * graph.face_points *
        graph.face_scalar_dofs);
    std::vector<double> face_weight(
        static_cast<std::size_t>(graph.elements) * 4 * graph.face_points);
    std::vector<double> face_normal(face_weight.size() * dimension);
    const double* old_values = graph.old_interior->HostRead();
    const double* current_values = graph.current_interior->HostRead();
    const double* trace_values = graph.current_trace->HostRead();

    for (int element = 0; element < graph.elements; ++element)
    {
      const mfem::FiniteElement* fe = graph.interior_space->GetFE(element);
      mfem::ElementTransformation* transformation =
          graph.mesh->GetElementTransformation(element);
      mfem::Array<int> interior_vdofs;
      graph.interior_space->GetElementVDofs(element, interior_vdofs);
      if (fe == nullptr || transformation == nullptr ||
          fe->GetDof() != graph.scalar_dofs ||
          interior_vdofs.Size() != graph.local_interior)
      {
        return false;
      }
      for (int local = 0; local < graph.local_interior; ++local)
      {
        double sign = 1.0;
        const int vdof = DecodeSignedVDof(interior_vdofs[local], sign);
        if (vdof < 0 || vdof >= graph.old_interior->Size()) { return false; }
        const std::size_t offset =
            static_cast<std::size_t>(element) * graph.local_interior + local;
        old_coefficients[offset] = sign * old_values[vdof];
        current_coefficients[offset] = sign * current_values[vdof];
      }
      const mfem::IntegrationRule& volume_rule =
          mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
      if (volume_rule.GetNPoints() != graph.volume_points) { return false; }
      mfem::Vector shape(graph.scalar_dofs), physical(dimension);
      mfem::DenseMatrix derivative(graph.scalar_dofs, dimension);
      for (int q = 0; q < graph.volume_points; ++q)
      {
        const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
        transformation->SetIntPoint(&point);
        transformation->Transform(point, physical);
        fe->CalcShape(point, shape);
        fe->CalcPhysDShape(*transformation, derivative);
        const std::size_t quadrature =
            static_cast<std::size_t>(element) * graph.volume_points + q;
        volume_weight[quadrature] = point.weight * transformation->Weight();
        for (int direction = 0; direction < dimension; ++direction)
        {
          volume_position[quadrature * dimension + direction] = physical[direction];
        }
        for (int scalar = 0; scalar < graph.scalar_dofs; ++scalar)
        {
          volume_shape[quadrature * graph.scalar_dofs + scalar] = shape[scalar];
          for (int direction = 0; direction < dimension; ++direction)
          {
            volume_derivative[(quadrature * graph.scalar_dofs + scalar) *
                                  dimension + direction] = derivative(scalar, direction);
          }
        }
      }

      mfem::Array<int> faces, orientations;
      graph.mesh->GetElementEdges(element, faces, orientations);
      if (faces.Size() != 4 || orientations.Size() != 4) { return false; }
      int trace_offset = 0;
      for (int local_face = 0; local_face < faces.Size(); ++local_face)
      {
        const int face = faces[local_face];
        mfem::FaceElementTransformations* face_transformation =
            graph.mesh->GetFaceElementTransformations(face);
        const mfem::FiniteElement* trace_fe = graph.trace_space->GetFaceElement(face);
        mfem::Array<int> face_vdofs;
        graph.trace_space->GetFaceVDofs(face, face_vdofs);
        if (face_transformation == nullptr || face_transformation->Face == nullptr ||
            trace_fe == nullptr || trace_fe->GetDof() != graph.face_scalar_dofs ||
            face_vdofs.Size() != conserved_components * graph.face_scalar_dofs)
        {
          return false;
        }
        const bool is_element_one = face_transformation->Elem1No == element;
        if (!is_element_one && face_transformation->Elem2No != element)
        {
          return false;
        }
        for (int local = 0; local < face_vdofs.Size(); ++local)
        {
          double sign = 1.0;
          const int vdof = DecodeSignedVDof(face_vdofs[local], sign);
          if (vdof < 0 || vdof >= graph.current_trace->Size()) { return false; }
          const std::size_t offset = static_cast<std::size_t>(element) *
                                         graph.local_trace +
                                     trace_offset + local;
          trace_coefficients[offset] = sign * trace_values[vdof];
          graph.signed_trace_dofs[offset] = face_vdofs[local];
        }
        const mfem::IntegrationRule& face_rule =
            mfem::IntRules.Get(trace_fe->GetGeomType(), quadrature_order);
        if (face_rule.GetNPoints() != graph.face_points) { return false; }
        mfem::Vector interior_shape(graph.scalar_dofs);
        mfem::Vector trace_shape(graph.face_scalar_dofs);
        mfem::Vector scaled_normal(dimension);
        for (int q = 0; q < graph.face_points; ++q)
        {
          const mfem::IntegrationPoint& face_point = face_rule.IntPoint(q);
          face_transformation->SetAllIntPoints(&face_point);
          const mfem::IntegrationPoint& element_point = is_element_one ?
              face_transformation->GetElement1IntPoint() :
              face_transformation->GetElement2IntPoint();
          fe->CalcShape(element_point, interior_shape);
          trace_fe->CalcShape(face_point, trace_shape);
          mfem::CalcOrtho(face_transformation->Face->Jacobian(), scaled_normal);
          const double measure = scaled_normal.Norml2();
          if (!(measure > 0.0) || !std::isfinite(measure)) { return false; }
          const double side = is_element_one ? 1.0 : -1.0;
          const std::size_t quadrature =
              (static_cast<std::size_t>(element) * 4 + local_face) *
                  graph.face_points + q;
          face_weight[quadrature] = face_point.weight * measure;
          face_normal[quadrature * dimension] = side * scaled_normal[0] / measure;
          face_normal[quadrature * dimension + 1] =
              side * scaled_normal[1] / measure;
          for (int scalar = 0; scalar < graph.scalar_dofs; ++scalar)
          {
            face_interior_shape[quadrature * graph.scalar_dofs + scalar] =
                interior_shape[scalar];
          }
          for (int scalar = 0; scalar < graph.face_scalar_dofs; ++scalar)
          {
            face_trace_shape[quadrature * graph.face_scalar_dofs + scalar] =
                trace_shape[scalar];
          }
        }
        trace_offset += face_vdofs.Size();
      }
      if (trace_offset != graph.local_trace) { return false; }
    }

    if (!AssignVector(storage.old_coefficients, old_coefficients) ||
        !AssignVector(storage.current_coefficients, current_coefficients) ||
        !AssignVector(storage.trace_coefficients, trace_coefficients) ||
        !AssignVector(storage.volume_shape, volume_shape) ||
        !AssignVector(storage.volume_derivative, volume_derivative) ||
        !AssignVector(storage.volume_weight, volume_weight) ||
        !AssignVector(storage.volume_position, volume_position) ||
        !AssignVector(storage.face_interior_shape, face_interior_shape) ||
        !AssignVector(storage.face_trace_shape, face_trace_shape) ||
        !AssignVector(storage.face_weight, face_weight) ||
        !AssignVector(storage.face_normal, face_normal))
    {
      return false;
    }
    const std::size_t batches = static_cast<std::size_t>(graph.elements);
    const std::size_t ni = static_cast<std::size_t>(graph.local_interior);
    const std::size_t nt = static_cast<std::size_t>(graph.local_trace);
    return SizeAndZero(storage.residual_r, batches * ni) &&
           SizeAndZero(storage.residual_g, batches * nt) &&
           SizeAndZero(storage.a, batches * ni * ni) &&
           SizeAndZero(storage.b, batches * ni * nt) &&
           SizeAndZero(storage.c, batches * nt * ni) &&
           SizeAndZero(storage.d, batches * nt * nt) &&
           SizeAndZero(storage.f, batches * ni) &&
           SizeAndZero(storage.g, batches * nt) &&
           SizeAndZero(storage.valid, batches) &&
           SizeAndZero(storage.candidate_residual_r, batches * ni) &&
           SizeAndZero(storage.candidate_residual_g, batches * nt) &&
           SizeAndZero(storage.candidate_a, batches * ni * ni) &&
           SizeAndZero(storage.candidate_b, batches * ni * nt) &&
           SizeAndZero(storage.candidate_c, batches * nt * ni) &&
           SizeAndZero(storage.candidate_d, batches * nt * nt) &&
           SizeAndZero(storage.candidate_f, batches * ni) &&
           SizeAndZero(storage.candidate_g, batches * nt) &&
           SizeAndZero(storage.candidate_valid, batches);
  }
  catch (const std::exception&)
  {
    return false;
  }
}

bool AssembleCase(const CaseGraph& graph, CaseStorage& storage,
                  const bool manufactured, const bool report_failure = true)
{
  const std::size_t batches = static_cast<std::size_t>(graph.elements);
  const std::size_t ni = static_cast<std::size_t>(graph.local_interior);
  const std::size_t nt = static_cast<std::size_t>(graph.local_trace);
  std::fill_n(storage.candidate_residual_r.HostWrite(), batches * ni, 0.0);
  std::fill_n(storage.candidate_residual_g.HostWrite(), batches * nt, 0.0);
  std::fill_n(storage.candidate_a.HostWrite(), batches * ni * ni, 0.0);
  std::fill_n(storage.candidate_b.HostWrite(), batches * ni * nt, 0.0);
  std::fill_n(storage.candidate_c.HostWrite(), batches * nt * ni, 0.0);
  std::fill_n(storage.candidate_d.HostWrite(), batches * nt * nt, 0.0);
  std::fill_n(storage.candidate_f.HostWrite(), batches * ni, 0.0);
  std::fill_n(storage.candidate_g.HostWrite(), batches * nt, 0.0);
  std::fill_n(storage.candidate_valid.HostWrite(), batches, 1.0);

  const double* old_coefficients = storage.old_coefficients.Read();
  const double* current_coefficients = storage.current_coefficients.Read();
  const double* trace_coefficients = storage.trace_coefficients.Read();
  const double* volume_shape = storage.volume_shape.Read();
  const double* volume_derivative = storage.volume_derivative.Read();
  const double* volume_weight = storage.volume_weight.Read();
  const double* volume_position = storage.volume_position.Read();
  const double* face_interior_shape = storage.face_interior_shape.Read();
  const double* face_trace_shape = storage.face_trace_shape.Read();
  const double* face_weight = storage.face_weight.Read();
  const double* face_normal = storage.face_normal.Read();
  double* residual_r = storage.candidate_residual_r.ReadWrite();
  double* residual_g = storage.candidate_residual_g.ReadWrite();
  double* a = storage.candidate_a.ReadWrite();
  double* b = storage.candidate_b.ReadWrite();
  double* c = storage.candidate_c.ReadWrite();
  double* d = storage.candidate_d.ReadWrite();
  double* f = storage.candidate_f.ReadWrite();
  double* g = storage.candidate_g.ReadWrite();
  double* valid = storage.candidate_valid.ReadWrite();
  const int elements = graph.elements;
  const int scalar_dofs = graph.scalar_dofs;
  const int face_scalar_dofs = graph.face_scalar_dofs;
  const int local_interior = graph.local_interior;
  const int local_trace = graph.local_trace;
  const int volume_points = graph.volume_points;
  const int face_points = graph.face_points;
  mfem::forall(elements, [=] MFEM_HOST_DEVICE (int element)
  {
    const std::size_t element_ni =
        static_cast<std::size_t>(element) * local_interior;
    const std::size_t element_nt =
        static_cast<std::size_t>(element) * local_trace;
    const std::size_t a_offset = element_ni * local_interior;
    const std::size_t b_offset = element_ni * local_trace;
    const std::size_t c_offset = element_nt * local_interior;
    const std::size_t d_offset = element_nt * local_trace;
    for (int q = 0; q < volume_points; ++q)
    {
      const std::size_t quadrature =
          static_cast<std::size_t>(element) * volume_points + q;
      const double weight = volume_weight[quadrature];
      if (!(weight > 0.0) || !isfinite(weight))
      {
        valid[element] = -1.0;
        return;
      }
      double state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
      double old_state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
      for (int scalar = 0; scalar < scalar_dofs; ++scalar)
      {
        const double shape = volume_shape[quadrature * scalar_dofs + scalar];
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = LocalByNodes(scalar, scalar_dofs, component);
          state[component] += shape * current_coefficients[element_ni + local];
          old_state[component] += shape * old_coefficients[element_ni + local];
        }
      }
      double source[conserved_components] = {0.0, 0.0, 0.0, 0.0};
      double flux_x[conserved_components];
      double flux_y[conserved_components];
      double jacobian_x[conserved_components * conserved_components];
      double jacobian_y[conserved_components * conserved_components];
      if (!FiniteState(state))
      {
        valid[element] = -2.0;
        return;
      }
      if (!FiniteState(old_state))
      {
        valid[element] = -2.1;
        return;
      }
      if (manufactured &&
          !ManufacturedSourceDevice(volume_position[quadrature * dimension],
                                    volume_position[quadrature * dimension + 1],
                                    source))
      {
        valid[element] = -2.2;
        return;
      }
      if (!FluxAndJacobian(state, 1.0, 0.0, flux_x, jacobian_x))
      {
        valid[element] = -2.3;
        return;
      }
      if (!FluxAndJacobian(state, 0.0, 1.0, flux_y, jacobian_y))
      {
        valid[element] = -2.4;
        return;
      }
      for (int row = 0; row < scalar_dofs; ++row)
      {
        const double row_shape =
            volume_shape[quadrature * scalar_dofs + row];
        const double derivative_x =
            volume_derivative[(quadrature * scalar_dofs + row) * dimension];
        const double derivative_y =
            volume_derivative[(quadrature * scalar_dofs + row) * dimension + 1];
        for (int row_component = 0; row_component < conserved_components;
             ++row_component)
        {
          const int local_row =
              LocalByNodes(row, scalar_dofs, row_component);
          residual_r[element_ni + local_row] += weight *
              (row_shape * (state[row_component] - old_state[row_component]) /
                   backward_euler_dt -
               derivative_x * flux_x[row_component] -
               derivative_y * flux_y[row_component] -
               row_shape * source[row_component]);
          for (int column = 0; column < scalar_dofs; ++column)
          {
            const double column_shape =
                volume_shape[quadrature * scalar_dofs + column];
            for (int column_component = 0;
                 column_component < conserved_components; ++column_component)
            {
              const int local_column =
                  LocalByNodes(column, scalar_dofs, column_component);
              const double mass = row_component == column_component ?
                  row_shape * column_shape / backward_euler_dt : 0.0;
              const double volume =
                  -(derivative_x *
                        jacobian_x[row_component * conserved_components +
                                   column_component] +
                    derivative_y *
                        jacobian_y[row_component * conserved_components +
                                   column_component]) *
                  column_shape;
              a[a_offset + static_cast<std::size_t>(local_row) *
                                   local_interior +
                local_column] += weight * (mass + volume);
            }
          }
        }
      }
    }

    for (int local_face = 0; local_face < 4; ++local_face)
    {
      const int trace_offset =
          local_face * conserved_components * face_scalar_dofs;
      for (int q = 0; q < face_points; ++q)
      {
        const std::size_t quadrature =
            (static_cast<std::size_t>(element) * 4 + local_face) * face_points + q;
        const double weight = face_weight[quadrature];
        double interior_state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
        double trace_state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
        for (int scalar = 0; scalar < scalar_dofs; ++scalar)
        {
          const double shape =
              face_interior_shape[quadrature * scalar_dofs + scalar];
          for (int component = 0; component < conserved_components; ++component)
          {
            interior_state[component] += shape *
                current_coefficients[element_ni +
                                     LocalByNodes(scalar, scalar_dofs, component)];
          }
        }
        for (int scalar = 0; scalar < face_scalar_dofs; ++scalar)
        {
          const double shape =
              face_trace_shape[quadrature * face_scalar_dofs + scalar];
          for (int component = 0; component < conserved_components; ++component)
          {
            trace_state[component] += shape *
                trace_coefficients[element_nt + trace_offset +
                                   LocalByNodes(scalar, face_scalar_dofs,
                                                component)];
          }
        }
        double trace_flux[conserved_components];
        double trace_jacobian[conserved_components * conserved_components];
        if (!(weight > 0.0) || !isfinite(weight))
        {
          valid[element] = -3.0;
          return;
        }
        if (!FiniteState(interior_state))
        {
          valid[element] = -4.0;
          return;
        }
        if (!FiniteState(trace_state))
        {
          valid[element] = -5.0;
          return;
        }
        if (!FluxAndJacobian(trace_state,
                             face_normal[quadrature * dimension],
                             face_normal[quadrature * dimension + 1],
                             trace_flux, trace_jacobian))
        {
          valid[element] = -6.0;
          return;
        }
        for (int row = 0; row < scalar_dofs; ++row)
        {
          const double row_shape =
              face_interior_shape[quadrature * scalar_dofs + row];
          for (int component = 0; component < conserved_components; ++component)
          {
            const int local_row = LocalByNodes(row, scalar_dofs, component);
            residual_r[element_ni + local_row] += weight * row_shape *
                (trace_flux[component] + fixed_scalar_tau *
                     (interior_state[component] - trace_state[component]));
          }
          for (int column = 0; column < scalar_dofs; ++column)
          {
            const double column_shape =
                face_interior_shape[quadrature * scalar_dofs + column];
            for (int component = 0; component < conserved_components; ++component)
            {
              const int local_row = LocalByNodes(row, scalar_dofs, component);
              const int local_column =
                  LocalByNodes(column, scalar_dofs, component);
              a[a_offset + static_cast<std::size_t>(local_row) *
                                   local_interior +
                local_column] += weight * row_shape * fixed_scalar_tau *
                                 column_shape;
            }
          }
          for (int column = 0; column < face_scalar_dofs; ++column)
          {
            const double column_shape =
                face_trace_shape[quadrature * face_scalar_dofs + column];
            for (int row_component = 0;
                 row_component < conserved_components; ++row_component)
              for (int column_component = 0;
                   column_component < conserved_components; ++column_component)
              {
                const int local_row =
                    LocalByNodes(row, scalar_dofs, row_component);
                const int local_column = trace_offset +
                    LocalByNodes(column, face_scalar_dofs, column_component);
                b[b_offset + static_cast<std::size_t>(local_row) * local_trace +
                  local_column] += weight * row_shape * column_shape *
                    (trace_jacobian[row_component * conserved_components +
                                    column_component] -
                     (row_component == column_component ? fixed_scalar_tau : 0.0));
              }
          }
        }
        for (int row = 0; row < face_scalar_dofs; ++row)
        {
          const double row_shape =
              face_trace_shape[quadrature * face_scalar_dofs + row];
          for (int component = 0; component < conserved_components; ++component)
          {
            const int local_row =
                trace_offset + LocalByNodes(row, face_scalar_dofs, component);
            residual_g[element_nt + local_row] += weight * row_shape *
                (trace_flux[component] + fixed_scalar_tau *
                     (interior_state[component] - trace_state[component]));
          }
          for (int column = 0; column < scalar_dofs; ++column)
          {
            const double column_shape =
                face_interior_shape[quadrature * scalar_dofs + column];
            for (int component = 0; component < conserved_components; ++component)
            {
              const int local_row =
                  trace_offset + LocalByNodes(row, face_scalar_dofs, component);
              const int local_column =
                  LocalByNodes(column, scalar_dofs, component);
              c[c_offset + static_cast<std::size_t>(local_row) * local_interior +
                local_column] += weight * row_shape * fixed_scalar_tau *
                                 column_shape;
            }
          }
          for (int column = 0; column < face_scalar_dofs; ++column)
          {
            const double column_shape =
                face_trace_shape[quadrature * face_scalar_dofs + column];
            for (int row_component = 0;
                 row_component < conserved_components; ++row_component)
              for (int column_component = 0;
                   column_component < conserved_components; ++column_component)
              {
                const int local_row = trace_offset +
                    LocalByNodes(row, face_scalar_dofs, row_component);
                const int local_column = trace_offset +
                    LocalByNodes(column, face_scalar_dofs, column_component);
                d[d_offset + static_cast<std::size_t>(local_row) * local_trace +
                  local_column] += weight * row_shape * column_shape *
                    (trace_jacobian[row_component * conserved_components +
                                    column_component] -
                     (row_component == column_component ? fixed_scalar_tau : 0.0));
              }
          }
        }
      }
    }
    for (int row = 0; row < local_interior; ++row)
    {
      f[element_ni + row] = -residual_r[element_ni + row];
    }
    for (int row = 0; row < local_trace; ++row)
    {
      g[element_nt + row] = -residual_g[element_nt + row];
    }
  });

  const double* validity = storage.candidate_valid.HostRead();
  for (int element = 0; element < graph.elements; ++element)
  {
    if (validity[element] != 1.0)
    {
      if (report_failure)
      {
        std::cerr << "HDG_EULER_DEVICE_INTERNAL_FAILURE stage=kernel_validity"
                  << " element=" << element
                  << " code=" << validity[element] << '\n';
      }
      return false;
    }
  }
  const bool finite_r = VectorFinite(storage.candidate_residual_r);
  const bool finite_g = VectorFinite(storage.candidate_residual_g);
  const bool finite_a = VectorFinite(storage.candidate_a);
  const bool finite_b = VectorFinite(storage.candidate_b);
  const bool finite_c = VectorFinite(storage.candidate_c);
  const bool finite_d = VectorFinite(storage.candidate_d);
  const bool finite_f = VectorFinite(storage.candidate_f);
  const bool finite_condensed_g = VectorFinite(storage.candidate_g);
  if (!finite_r || !finite_g || !finite_a || !finite_b || !finite_c ||
      !finite_d || !finite_f || !finite_condensed_g)
  {
    if (report_failure)
    {
      std::cerr << "HDG_EULER_DEVICE_INTERNAL_FAILURE stage=kernel_finite"
                << " r=" << finite_r << " g=" << finite_g
                << " a=" << finite_a << " b=" << finite_b
                << " c=" << finite_c << " d=" << finite_d
                << " f=" << finite_f
                << " condensed_g=" << finite_condensed_g << '\n';
    }
    return false;
  }
  storage.residual_r.Swap(storage.candidate_residual_r);
  storage.residual_g.Swap(storage.candidate_residual_g);
  storage.a.Swap(storage.candidate_a);
  storage.b.Swap(storage.candidate_b);
  storage.c.Swap(storage.candidate_c);
  storage.d.Swap(storage.candidate_d);
  storage.f.Swap(storage.candidate_f);
  storage.g.Swap(storage.candidate_g);
  storage.valid.Swap(storage.candidate_valid);
  ++storage.generation;
  return true;
}

std::vector<double> Values(const mfem::Vector& vector)
{
  const double* values = vector.HostRead();
  return std::vector<double>(values, values + vector.Size());
}

double MaxScaledDifference(const double* actual, const std::size_t size,
                           const std::vector<double>& reference)
{
  if (reference.size() != size)
  {
    return std::numeric_limits<double>::infinity();
  }
  double error = 0.0;
  for (std::size_t entry = 0; entry < size; ++entry)
  {
    if (!std::isfinite(actual[entry]) || !std::isfinite(reference[entry]))
    {
      return std::numeric_limits<double>::infinity();
    }
    error = std::max(error, std::abs(actual[entry] - reference[entry]) /
                              std::max(1.0, std::abs(reference[entry])));
  }
  return error;
}

double MaxScaledDifference(const std::vector<double>& actual,
                           const std::vector<double>& reference)
{
  return MaxScaledDifference(actual.data(), actual.size(), reference);
}

double MaxScaledDifference(const mfem::Vector& actual,
                           const std::vector<double>& reference)
{
  return MaxScaledDifference(actual.HostRead(),
                             static_cast<std::size_t>(actual.Size()), reference);
}

double MaxAbs(const mfem::Vector& vector)
{
  double result = 0.0;
  const double* values = vector.HostRead();
  for (int entry = 0; entry < vector.Size(); ++entry)
  {
    result = std::max(result, std::abs(values[entry]));
  }
  return result;
}

bool SameValues(const mfem::Vector& vector,
                const std::vector<double>& saved)
{
  if (vector.Size() != static_cast<int>(saved.size())) { return false; }
  const double* values = vector.HostRead();
  return std::equal(saved.begin(), saved.end(), values);
}

struct PublishedStorageSnapshot
{
  std::vector<double> residual_r, residual_g, a, b, c, d, f, g, valid;
  std::uint64_t generation = 0;
};

PublishedStorageSnapshot CapturePublishedStorage(const CaseStorage& storage)
{
  return {Values(storage.residual_r), Values(storage.residual_g),
          Values(storage.a), Values(storage.b), Values(storage.c),
          Values(storage.d), Values(storage.f), Values(storage.g),
          Values(storage.valid), storage.generation};
}

bool SamePublishedStorage(const CaseStorage& storage,
                          const PublishedStorageSnapshot& saved)
{
  return storage.generation == saved.generation &&
         SameValues(storage.residual_r, saved.residual_r) &&
         SameValues(storage.residual_g, saved.residual_g) &&
         SameValues(storage.a, saved.a) && SameValues(storage.b, saved.b) &&
         SameValues(storage.c, saved.c) && SameValues(storage.d, saved.d) &&
         SameValues(storage.f, saved.f) && SameValues(storage.g, saved.g) &&
         SameValues(storage.valid, saved.valid);
}

bool RunPhysicsDeviceCheck(double& finite_difference_error,
                           double& hand_flux_error)
{
  mfem::Vector results(2);
  results.UseDevice(true);
  std::fill_n(results.HostWrite(), 2, 0.0);
  double* output = results.ReadWrite();
  mfem::forall(1, [=] MFEM_HOST_DEVICE (int)
  {
    const double state[conserved_components] = {1.2, 0.48, -0.3, 2.3835};
    const double expected[conserved_components] =
        {0.048, 0.5592, 0.708, 0.13134};
    double flux[conserved_components];
    double jacobian[conserved_components * conserved_components];
    if (!FluxAndJacobian(state, 0.6, 0.8, flux, jacobian))
    {
      output[0] = 1.0e300;
      output[1] = 1.0e300;
      return;
    }
    double hand_error = 0.0;
    for (int row = 0; row < conserved_components; ++row)
    {
      const double error = fabs(flux[row] - expected[row]);
      hand_error = error > hand_error ? error : hand_error;
    }
    double fd_error = 0.0;
    for (int column = 0; column < conserved_components; ++column)
    {
      double plus[conserved_components];
      double minus[conserved_components];
      for (int entry = 0; entry < conserved_components; ++entry)
      {
        plus[entry] = state[entry];
        minus[entry] = state[entry];
      }
      const double step = 1.0e-7 *
          (fabs(state[column]) > 1.0 ? fabs(state[column]) : 1.0);
      plus[column] += step;
      minus[column] -= step;
      double plus_flux[conserved_components];
      double minus_flux[conserved_components];
      double unused[conserved_components * conserved_components];
      if (!FluxAndJacobian(plus, 0.6, 0.8, plus_flux, unused) ||
          !FluxAndJacobian(minus, 0.6, 0.8, minus_flux, unused))
      {
        output[0] = 1.0e300;
        output[1] = 1.0e300;
        return;
      }
      for (int row = 0; row < conserved_components; ++row)
      {
        const double finite_difference =
            (plus_flux[row] - minus_flux[row]) / (2.0 * step);
        const double scale = fabs(jacobian[row * conserved_components + column]);
        const double error = fabs(finite_difference -
                                  jacobian[row * conserved_components + column]) /
                             (scale > 1.0 ? scale : 1.0);
        fd_error = error > fd_error ? error : fd_error;
      }
    }
    output[0] = fd_error;
    output[1] = hand_error;
  });
  const double* host = results.HostRead();
  finite_difference_error = host[0];
  hand_flux_error = host[1];
  return std::isfinite(finite_difference_error) &&
         std::isfinite(hand_flux_error) &&
         finite_difference_error <= physics_gate &&
         hand_flux_error <= constant_gate;
}

bool RunConstantAndFailureAtomicCheck(double& constant_residual)
{
  CaseGraph graph;
  CaseStorage storage;
  if (!BuildCase(1, 3, false, graph, storage))
  {
    std::cerr << "HDG_EULER_DEVICE_INTERNAL_FAILURE stage=constant_build\n";
    return false;
  }
  if (!AssembleCase(graph, storage, false))
  {
    std::cerr << "HDG_EULER_DEVICE_INTERNAL_FAILURE stage=constant_assemble\n";
    return false;
  }
  constant_residual = MaxAbs(storage.residual_r);
  std::vector<double> assembled_trace(
      static_cast<std::size_t>(graph.trace_space->GetVSize()), 0.0);
  const double* local_g = storage.residual_g.HostRead();
  for (int element = 0; element < graph.elements; ++element)
    for (int row = 0; row < graph.local_trace; ++row)
    {
      double sign = 1.0;
      const std::size_t local = static_cast<std::size_t>(element) *
                                    graph.local_trace +
                                row;
      const int vdof = DecodeSignedVDof(graph.signed_trace_dofs[local], sign);
      if (vdof < 0 || vdof >= graph.trace_space->GetVSize()) { return false; }
      assembled_trace[static_cast<std::size_t>(vdof)] += sign * local_g[local];
    }
  for (const double value : assembled_trace)
  {
    constant_residual = std::max(constant_residual, std::abs(value));
  }
  if (!(constant_residual <= constant_gate))
  {
    std::cerr << "HDG_EULER_DEVICE_INTERNAL_FAILURE stage=constant_gate value="
              << constant_residual << '\n';
    return false;
  }

  const PublishedStorageSnapshot saved = CapturePublishedStorage(storage);
  double* coefficients = storage.current_coefficients.HostReadWrite();
  const double original = coefficients[0];
  coefficients[0] = std::numeric_limits<double>::quiet_NaN();
  const bool nonfinite_rejected = !AssembleCase(graph, storage, false, false);
  coefficients = storage.current_coefficients.HostReadWrite();
  coefficients[0] = original;
  if (!nonfinite_rejected || !SamePublishedStorage(storage, saved))
  {
    std::cerr << "HDG_EULER_DEVICE_INTERNAL_FAILURE stage=nonfinite_atomic"
              << " rejected=" << nonfinite_rejected
              << " generation=" << storage.generation
              << " expected_generation=" << saved.generation << '\n';
    return false;
  }

  double* weights = storage.volume_weight.HostReadWrite();
  const double original_weight = weights[0];
  weights[0] = -1.0;
  const bool invalid_rejected = !AssembleCase(graph, storage, false, false);
  weights = storage.volume_weight.HostReadWrite();
  weights[0] = original_weight;
  if (!invalid_rejected || !SamePublishedStorage(storage, saved) ||
      !AssembleCase(graph, storage, false) ||
      storage.generation != saved.generation + 1)
  {
    std::cerr << "HDG_EULER_DEVICE_INTERNAL_FAILURE stage=invalid_atomic"
              << " rejected=" << invalid_rejected
              << " generation=" << storage.generation
              << " expected_generation=" << saved.generation + 1 << '\n';
    return false;
  }
  return true;
}

bool BuildReferenceCase(const int p, const int N, const bool full_trace_flow,
                        ReferenceCase& output)
{
  output = {};
  SmoothEulerExperiment reference(p);
  if (!reference.BuildPeriodicMesh(N) || !reference.ProjectOldStage() ||
      !reference.AssembleLinearization() ||
      !reference.CopyLocalLinearization(output.local))
  {
    return false;
  }
  if (!full_trace_flow) { return true; }
  if (!reference.CondenseAndAssembleTrace() || !reference.SolveTrace() ||
      !reference.CopyTraceRightHandSide(output.trace_rhs) ||
      !reference.CopyTraceIncrement(output.trace_solution))
  {
    return false;
  }
  const mfem::SparseMatrix* matrix = reference.trace_matrix();
  if (matrix == nullptr || matrix->Height() <= 0 ||
      matrix->Height() != matrix->Width() ||
      output.trace_rhs.size() != static_cast<std::size_t>(matrix->Height()) ||
      output.trace_solution.size() !=
          static_cast<std::size_t>(matrix->Height()))
  {
    return false;
  }
  output.global_trace_dofs = matrix->Height();
  output.trace_matrix.resize(
      static_cast<std::size_t>(matrix->Height()) * matrix->Width());
  const mfem::SparseMatrix& assembled = *matrix;
  for (int row = 0; row < matrix->Height(); ++row)
    for (int column = 0; column < matrix->Width(); ++column)
    {
      output.trace_matrix[static_cast<std::size_t>(row) * matrix->Width() +
                          column] = assembled(row, column);
    }
  return true;
}

struct ReferenceRepository
{
  std::array<ReferenceCase, 4> cases;
  bool ready = false;
};

ReferenceRepository& References()
{
  static ReferenceRepository repository;
  return repository;
}

bool AssembleTraceSystem(const CaseGraph& graph,
                         const std::vector<double>& local_schur,
                         const std::vector<double>& local_rhs,
                         std::unique_ptr<mfem::SparseMatrix>& matrix,
                         mfem::Vector& rhs)
{
  const int global_dofs = graph.trace_space->GetVSize();
  if (local_schur.size() != static_cast<std::size_t>(graph.elements) *
                                 graph.local_trace * graph.local_trace ||
      local_rhs.size() != static_cast<std::size_t>(graph.elements) *
                              graph.local_trace ||
      graph.signed_trace_dofs.size() != local_rhs.size() || global_dofs <= 0)
  {
    return false;
  }
  try
  {
    auto candidate = std::make_unique<mfem::SparseMatrix>(global_dofs);
    rhs.SetSize(global_dofs);
    rhs = 0.0;
    for (int element = 0; element < graph.elements; ++element)
    {
      mfem::Array<int> dofs(graph.local_trace);
      mfem::DenseMatrix local(graph.local_trace);
      for (int row = 0; row < graph.local_trace; ++row)
      {
        const std::size_t row_offset =
            static_cast<std::size_t>(element) * graph.local_trace + row;
        dofs[row] = graph.signed_trace_dofs[row_offset];
        double sign = 1.0;
        const int vdof = DecodeSignedVDof(dofs[row], sign);
        if (vdof < 0 || vdof >= global_dofs) { return false; }
        rhs[vdof] += sign * local_rhs[row_offset];
        for (int column = 0; column < graph.local_trace; ++column)
        {
          local(row, column) = local_schur[
              static_cast<std::size_t>(element) * graph.local_trace *
                  graph.local_trace +
              static_cast<std::size_t>(row) * graph.local_trace + column];
        }
      }
      candidate->AddSubMatrix(dofs, dofs, local);
    }
    candidate->Finalize();
    matrix = std::move(candidate);
  }
  catch (const std::exception&)
  {
    return false;
  }
  return true;
}

bool SolveTraceSystem(const mfem::SparseMatrix& matrix, const mfem::Vector& rhs,
                      mfem::Vector& solution, double& true_residual)
{
  if (matrix.Height() <= 0 || matrix.Height() != matrix.Width() ||
      rhs.Size() != matrix.Height())
  {
    return false;
  }
  try
  {
    mfem::DenseMatrix dense(matrix.Height());
    for (int row = 0; row < matrix.Height(); ++row)
      for (int column = 0; column < matrix.Width(); ++column)
      {
        dense(row, column) = matrix(row, column);
      }
    mfem::DenseMatrixInverse inverse(dense);
    solution.SetSize(rhs.Size());
    inverse.Mult(rhs, solution);
    mfem::Vector residual(rhs.Size());
    matrix.Mult(solution, residual);
    residual -= rhs;
    true_residual = residual.Norml2() / std::max(rhs.Norml2(), 1.0e-300);
  }
  catch (const std::exception&)
  {
    return false;
  }
  return std::isfinite(true_residual) && true_residual <= true_residual_gate;
}

StridedBlockView<const double> PackedConstView(const double* data,
                                               const LocalIndex batches,
                                               const LocalIndex rows,
                                               const LocalIndex columns) noexcept
{
  return {data, batches, rows, columns, rows * columns, columns, 1};
}

core::LocalSystemBatchView StorageInput(const CaseGraph& graph,
                                        const CaseStorage& storage)
{
  const LocalIndex batches = graph.elements;
  const LocalIndex ni = graph.local_interior;
  const LocalIndex nt = graph.local_trace;
  return {{ni, nt, 1, 1},
          PackedConstView(storage.a.HostRead(), batches, ni, ni),
          PackedConstView(storage.b.HostRead(), batches, ni, nt),
          PackedConstView(storage.c.HostRead(), batches, nt, ni),
          PackedConstView(storage.d.HostRead(), batches, nt, nt),
          PackedConstView(storage.f.HostRead(), batches, ni, 1),
          PackedConstView(storage.g.HostRead(), batches, nt, 1)};
}

core::LocalSystemBatchView SnapshotInput(const LocalLinearizationSnapshot& snapshot)
{
  const LocalIndex batches = snapshot.elements;
  const LocalIndex ni = snapshot.interior_dofs;
  const LocalIndex nt = snapshot.trace_dofs;
  return {{ni, nt, 1, 1},
          PackedConstView(snapshot.a.data(), batches, ni, ni),
          PackedConstView(snapshot.b.data(), batches, ni, nt),
          PackedConstView(snapshot.c.data(), batches, nt, ni),
          PackedConstView(snapshot.d.data(), batches, nt, nt),
          PackedConstView(snapshot.f.data(), batches, ni, 1),
          PackedConstView(snapshot.g.data(), batches, nt, 1)};
}

bool FullTraceFlow(const CaseGraph& graph, const ReferenceCase& reference,
                   device::DeviceCoreEngine& engine,
                   core::HostReferenceEngine& oracle,
                   const std::vector<double>& schur,
                   const std::vector<double>& condensed_rhs,
                   DeviceEulerCaseEvidence& evidence)
{
  std::unique_ptr<mfem::SparseMatrix> matrix;
  mfem::Vector rhs;
  if (!AssembleTraceSystem(graph, schur, condensed_rhs, matrix, rhs) || !matrix)
  {
    evidence.status = DeviceEulerStatus::trace_assembly_failure;
    return false;
  }
  mfem::Vector trace_solution;
  if (!SolveTraceSystem(*matrix, rhs, trace_solution, evidence.true_residual))
  {
    evidence.status = DeviceEulerStatus::trace_solve_failure;
    return false;
  }
  if (reference.global_trace_dofs != matrix->Height() ||
      reference.trace_matrix.size() !=
          static_cast<std::size_t>(matrix->Height()) * matrix->Width() ||
      reference.trace_rhs.size() != static_cast<std::size_t>(matrix->Height()) ||
      reference.trace_solution.size() !=
          static_cast<std::size_t>(matrix->Height()))
  {
    evidence.status = DeviceEulerStatus::oracle_failure;
    return false;
  }
  evidence.max_trace_solution_parity_error =
      MaxScaledDifference(trace_solution, reference.trace_solution);
  evidence.max_condensed_parity_error = std::max(
      evidence.max_condensed_parity_error,
      MaxScaledDifference(rhs, reference.trace_rhs));
  const mfem::SparseMatrix& assembled_matrix = *matrix;
  for (int row = 0; row < matrix->Height(); ++row)
    for (int column = 0; column < matrix->Width(); ++column)
    {
      const double expected = reference.trace_matrix[
          static_cast<std::size_t>(row) * matrix->Width() + column];
      const double error = std::abs(assembled_matrix(row, column) - expected) /
                           std::max(1.0, std::abs(expected));
      evidence.max_condensed_parity_error =
          std::max(evidence.max_condensed_parity_error, error);
    }

  std::vector<double> local_trace(
      static_cast<std::size_t>(graph.elements) * graph.local_trace);
  for (int element = 0; element < graph.elements; ++element)
    for (int row = 0; row < graph.local_trace; ++row)
    {
      double sign = 1.0;
      const std::size_t local = static_cast<std::size_t>(element) *
                                    graph.local_trace +
                                row;
      const int vdof = DecodeSignedVDof(graph.signed_trace_dofs[local], sign);
      if (vdof < 0 || vdof >= trace_solution.Size())
      {
        evidence.status = DeviceEulerStatus::recovery_failure;
        return false;
      }
      local_trace[local] = sign * trace_solution[vdof];
    }
  std::vector<double> device_recovery(
      static_cast<std::size_t>(graph.elements) * graph.local_interior);
  std::vector<double> reference_recovery(device_recovery.size());
  const auto trace_view = PackedConstView(local_trace.data(), graph.elements,
                                          graph.local_trace, 1);
  if (engine.Recover(trace_view,
                     View(device_recovery.data(), graph.elements,
                          graph.local_interior, 1)) != core::Status::success ||
      oracle.Recover(trace_view,
                     View(reference_recovery.data(), graph.elements,
                          graph.local_interior, 1)) != core::Status::success)
  {
    evidence.status = DeviceEulerStatus::recovery_failure;
    return false;
  }
  evidence.max_recovery_parity_error = std::max(
      evidence.max_recovery_parity_error,
      MaxScaledDifference(device_recovery, reference_recovery));
  evidence.full_trace_flow = true;
  return true;
}

bool RunCase(const int p, const int N, const double physics_error,
             const double hand_flux_error, const double constant_residual,
             const ReferenceCase& reference,
             DeviceEulerCaseEvidence& evidence)
{
  evidence = {};
  evidence.p = p;
  evidence.N = N;
  evidence.max_physics_finite_difference_error = physics_error;
  evidence.max_hand_flux_error = hand_flux_error;
  evidence.constant_residual = constant_residual;

  CaseGraph graph;
  CaseStorage storage;
  if (!BuildCase(p, N, true, graph, storage))
  {
    evidence.status = DeviceEulerStatus::projection_failure;
    return false;
  }
  if (!AssembleCase(graph, storage, true))
  {
    evidence.status = DeviceEulerStatus::assembly_failure;
    return false;
  }
  evidence.generation = storage.generation;
  evidence.elements = graph.elements;
  evidence.interior_unknowns = graph.local_interior;
  evidence.trace_unknowns = graph.local_trace;

  const LocalLinearizationSnapshot& snapshot = reference.local;
  if (
      snapshot.elements != graph.elements ||
      snapshot.interior_dofs != graph.local_interior ||
      snapshot.trace_dofs != graph.local_trace ||
      snapshot.signed_trace_dofs != graph.signed_trace_dofs)
  {
    evidence.status = DeviceEulerStatus::oracle_failure;
    return false;
  }
  evidence.max_residual_parity_error = std::max(
      MaxScaledDifference(storage.f, snapshot.f),
      MaxScaledDifference(storage.g, snapshot.g));
  evidence.max_jacobian_parity_error = std::max(
      std::max(MaxScaledDifference(storage.a, snapshot.a),
               MaxScaledDifference(storage.b, snapshot.b)),
      std::max(MaxScaledDifference(storage.c, snapshot.c),
               MaxScaledDifference(storage.d, snapshot.d)));

  const std::size_t batches = static_cast<std::size_t>(graph.elements);
  const std::size_t ni = static_cast<std::size_t>(graph.local_interior);
  const std::size_t nt = static_cast<std::size_t>(graph.local_trace);
  std::vector<double> device_schur(batches * nt * nt);
  std::vector<double> device_rhs(batches * nt);
  std::vector<double> reference_schur(device_schur.size());
  std::vector<double> reference_rhs(device_rhs.size());
  device::DeviceCoreEngine engine;
  core::HostReferenceEngine oracle;
  const core::LocalSystemBatchView device_input = StorageInput(graph, storage);
  core::LocalSystemBatchView reference_input = SnapshotInput(snapshot);
  if (engine.Condense(device_input,
                      {View(device_schur.data(), graph.elements,
                            graph.local_trace, graph.local_trace),
                       View(device_rhs.data(), graph.elements,
                            graph.local_trace, 1)}) != core::Status::success ||
      oracle.Condense(reference_input,
                      {View(reference_schur.data(), graph.elements,
                            graph.local_trace, graph.local_trace),
                       View(reference_rhs.data(), graph.elements,
                            graph.local_trace, 1)}) != core::Status::success)
  {
    evidence.status = DeviceEulerStatus::condensation_failure;
    return false;
  }
  evidence.max_condensed_parity_error = std::max(
      MaxScaledDifference(device_schur, reference_schur),
      MaxScaledDifference(device_rhs, reference_rhs));

  std::vector<double> trace(batches * nt);
  for (std::size_t entry = 0; entry < trace.size(); ++entry)
  {
    trace[entry] = 1.0e-3 * std::sin(0.25 * static_cast<double>(entry + 1));
  }
  std::vector<double> device_action(trace.size()), reference_action(trace.size());
  std::vector<double> device_recovery(batches * ni), reference_recovery(batches * ni);
  const auto trace_view = PackedConstView(trace.data(), graph.elements,
                                          graph.local_trace, 1);
  if (engine.ApplySchur(trace_view,
                        View(device_action.data(), graph.elements,
                             graph.local_trace, 1)) != core::Status::success ||
      oracle.ApplySchur(trace_view,
                        View(reference_action.data(), graph.elements,
                             graph.local_trace, 1)) != core::Status::success ||
      engine.Recover(trace_view,
                     View(device_recovery.data(), graph.elements,
                          graph.local_interior, 1)) != core::Status::success ||
      oracle.Recover(trace_view,
                     View(reference_recovery.data(), graph.elements,
                          graph.local_interior, 1)) != core::Status::success)
  {
    evidence.status = DeviceEulerStatus::recovery_failure;
    return false;
  }
  evidence.max_schur_action_parity_error =
      MaxScaledDifference(device_action, reference_action);
  evidence.max_recovery_parity_error =
      MaxScaledDifference(device_recovery, reference_recovery);

  const std::size_t allocation_events = engine.accounting().allocation_events;
  for (int repeat = 0; repeat < 2; ++repeat)
  {
    if (engine.ApplySchur(trace_view,
                          View(device_action.data(), graph.elements,
                               graph.local_trace, 1)) != core::Status::success ||
        engine.Recover(trace_view,
                       View(device_recovery.data(), graph.elements,
                            graph.local_interior, 1)) != core::Status::success ||
        engine.accounting().allocation_events != allocation_events ||
        engine.accounting().steady_state_allocation_events != 0)
    {
      evidence.status = DeviceEulerStatus::recovery_failure;
      return false;
    }
  }

  const device::ExecutionPath stable_path = engine.execution_path();
  std::vector<double> bad_trace = trace;
  bad_trace[0] = std::numeric_limits<double>::quiet_NaN();
  std::fill(device_action.begin(), device_action.end(), 73.0);
  const std::vector<double> saved_action = device_action;
  if (engine.ApplySchur(
          PackedConstView(bad_trace.data(), graph.elements, graph.local_trace, 1),
          View(device_action.data(), graph.elements, graph.local_trace, 1)) !=
          core::Status::non_finite ||
      device_action != saved_action || engine.execution_path() != stable_path ||
      engine.accounting().allocation_events != allocation_events)
  {
    evidence.status = DeviceEulerStatus::non_finite;
    return false;
  }
  std::vector<double> bad_f = Values(storage.f);
  bad_f[0] = std::numeric_limits<double>::infinity();
  core::LocalSystemBatchView bad_input = device_input;
  bad_input.f = PackedConstView(bad_f.data(), graph.elements,
                                graph.local_interior, 1);
  std::vector<double> rejected_schur(device_schur.size(), 79.0);
  std::vector<double> rejected_rhs(device_rhs.size(), 83.0);
  const std::vector<double> saved_rejected_schur = rejected_schur;
  const std::vector<double> saved_rejected_rhs = rejected_rhs;
  if (engine.Condense(
          bad_input,
          {View(rejected_schur.data(), graph.elements, graph.local_trace,
                graph.local_trace),
           View(rejected_rhs.data(), graph.elements, graph.local_trace, 1)}) !=
          core::Status::non_finite ||
      rejected_schur != saved_rejected_schur ||
      rejected_rhs != saved_rejected_rhs ||
      engine.execution_path() != stable_path ||
      engine.accounting().allocation_events != allocation_events)
  {
    evidence.status = DeviceEulerStatus::non_finite;
    return false;
  }

  if (p == 1 && N == 3 &&
      !FullTraceFlow(graph, reference, engine, oracle, device_schur, device_rhs,
                     evidence))
  {
    return false;
  }
  evidence.execution_path = engine.execution_path();
  evidence.accounting = engine.accounting();
  bool execution_ok = false;
#if defined(MFEM_USE_CUDA)
  execution_ok = evidence.execution_path == device::ExecutionPath::mfem_device &&
                 engine.using_device() &&
                 evidence.accounting.host_reference_fallbacks == 0 &&
                 evidence.accounting.persistent_bytes > 0 &&
                 evidence.accounting.allocation_events > 0 &&
                 evidence.accounting.host_to_device_bytes > 0 &&
                 evidence.accounting.device_to_host_bytes > 0 &&
                 evidence.accounting.steady_state_allocation_events == 0;
#else
  execution_ok = evidence.execution_path == device::ExecutionPath::host_reference &&
                 !engine.using_device() &&
                 evidence.accounting.host_reference_fallbacks > 0;
#endif
  const bool trace_flow_ok = p == 1 && N == 3 ?
      evidence.full_trace_flow &&
          evidence.max_trace_solution_parity_error <= solve_gate &&
          evidence.true_residual <= true_residual_gate :
      !evidence.full_trace_flow &&
          std::isinf(evidence.max_trace_solution_parity_error) &&
          std::isinf(evidence.true_residual);
  evidence.pass = execution_ok && evidence.generation > 0 &&
      evidence.max_physics_finite_difference_error <= physics_gate &&
      evidence.max_hand_flux_error <= constant_gate &&
      evidence.constant_residual <= constant_gate &&
      evidence.max_residual_parity_error <= local_gate &&
      evidence.max_jacobian_parity_error <= local_gate &&
      evidence.max_condensed_parity_error <= local_gate &&
      evidence.max_schur_action_parity_error <= solve_gate &&
      evidence.max_recovery_parity_error <= solve_gate &&
      trace_flow_ok;
  evidence.status = evidence.pass ? DeviceEulerStatus::success :
                                    DeviceEulerStatus::oracle_failure;
  return evidence.pass;
}
}  // namespace

namespace detail
{
bool PrepareSmoothEulerDeviceOracles()
{
  ReferenceRepository& repository = References();
  if (repository.ready) { return true; }
  if (mfem::Device::IsConfigured()) { return false; }

  std::array<ReferenceCase, 4> prepared;
  for (int p = 1; p <= 4; ++p)
  {
    if (!BuildReferenceCase(p, 3, p == 1,
                            prepared[static_cast<std::size_t>(p - 1)]))
    {
      return false;
    }
  }
  repository.cases = std::move(prepared);
  repository.ready = true;
  return true;
}
}  // namespace detail

bool RunSmoothEulerDeviceExperiment(DeviceEulerEvidence& evidence)
{
  DeviceEulerEvidence next;
  const ReferenceRepository& repository = References();
  if (!mfem::Device::IsConfigured() || !repository.ready)
  {
    next.status = DeviceEulerStatus::setup_failure;
    evidence = std::move(next);
    return false;
  }
  double physics_error = std::numeric_limits<double>::infinity();
  double hand_flux_error = std::numeric_limits<double>::infinity();
  double constant_residual = std::numeric_limits<double>::infinity();
  if (!RunPhysicsDeviceCheck(physics_error, hand_flux_error))
  {
    std::cerr << "HDG_EULER_DEVICE_INTERNAL_FAILURE stage=physics"
              << " fd=" << physics_error
              << " hand_flux=" << hand_flux_error << '\n';
    next.status = DeviceEulerStatus::oracle_failure;
    evidence = std::move(next);
    return false;
  }
  if (!RunConstantAndFailureAtomicCheck(constant_residual))
  {
    next.status = DeviceEulerStatus::oracle_failure;
    evidence = std::move(next);
    return false;
  }
  next.cases.reserve(4);
  for (int p = 1; p <= 4; ++p)
  {
    DeviceEulerCaseEvidence case_evidence;
    const bool passed = RunCase(p, 3, physics_error, hand_flux_error,
                                constant_residual,
                                repository.cases[static_cast<std::size_t>(p - 1)],
                                case_evidence);
    next.cases.push_back(std::move(case_evidence));
    if (!passed)
    {
      next.status = next.cases.back().status;
      evidence = std::move(next);
      return false;
    }
  }
  next.status = DeviceEulerStatus::success;
  next.pass = true;
  evidence = std::move(next);
  return true;
}
}  // namespace dg::hdg::euler

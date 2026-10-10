#include "numerics/hdg/trace/trace_dg_mpi.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

using dg::hdg::GlobalIndex;
using dg::hdg::Ownership;
using dg::hdg::trace::BuildDGInterfaceTraceExchangePlan;
using dg::hdg::trace::BuildDGInterfaceTraceMap;
using dg::hdg::trace::DGInterfaceFaceMap;
using dg::hdg::trace::DGInterfaceTraceExchangePlan;
using dg::hdg::trace::DGInterfaceTraceMap;
using dg::hdg::trace::TraceMapEntry;

constexpr int vector_dimension = 4;

int DecodeSignedVDof(const int signed_vdof)
{
  if (signed_vdof == std::numeric_limits<int>::min()) { return -1; }
  return signed_vdof < 0 ? -1 - signed_vdof : signed_vdof;
}

int SignedVDofSign(const int signed_vdof)
{
  return signed_vdof < 0 ? -1 : 1;
}

bool NearlyEqual(const double left, const double right)
{
  const double scale = 1.0 + std::max(std::abs(left), std::abs(right));
  return std::abs(left - right) <= 1.0e-12 * scale;
}

bool SameValues(const mfem::Vector& left, const mfem::Vector& right)
{
  if (left.Size() != right.Size()) { return false; }
  const double* const left_data = left.HostRead();
  const double* const right_data = right.HostRead();
  for (int i = 0; i < left.Size(); ++i)
  {
    if (left_data[i] != right_data[i]) { return false; }
  }
  return true;
}

bool SameTraceMap(const DGInterfaceTraceMap& left,
                  const DGInterfaceTraceMap& right)
{
  if (left.vector_dimension != right.vector_dimension ||
      left.ordering != right.ordering ||
      left.dofs.rank != right.dofs.rank ||
      left.dofs.rank_count != right.dofs.rank_count ||
      left.dofs.local_true_dof_count != right.dofs.local_true_dof_count ||
      left.dofs.global_true_dof_count != right.dofs.global_true_dof_count ||
      left.dofs.entries.size() != right.dofs.entries.size() ||
      left.faces.size() != right.faces.size())
  {
    return false;
  }
  for (std::size_t i = 0; i < left.dofs.entries.size(); ++i)
  {
    const TraceMapEntry& a = left.dofs.entries[i];
    const TraceMapEntry& b = right.dofs.entries[i];
    if (a.local_dof != b.local_dof ||
        a.global_true_dof != b.global_true_dof ||
        a.local_true_dof != b.local_true_dof ||
        a.owner_rank != b.owner_rank || a.ownership != b.ownership)
    {
      return false;
    }
  }
  for (std::size_t i = 0; i < left.faces.size(); ++i)
  {
    if (left.faces[i].local_face != right.faces[i].local_face ||
        left.faces[i].geometry != right.faces[i].geometry ||
        left.faces[i].signed_local_vdofs != right.faces[i].signed_local_vdofs)
    {
      return false;
    }
  }
  return true;
}

DGInterfaceTraceMap SentinelMap()
{
  DGInterfaceTraceMap map;
  map.vector_dimension = 37;
  map.ordering = mfem::Ordering::byVDIM;
  map.dofs.rank = 41;
  map.dofs.rank_count = 43;
  map.dofs.local_true_dof_count = 47;
  map.dofs.global_true_dof_count = 53;
  map.dofs.entries.push_back({59, 61, 67, 71, Ownership::ghost});
  map.faces.push_back(
      {73, mfem::Geometry::TRIANGLE, std::vector<int>{79, -83}});
  return map;
}

bool GlobalAll(MPI_Comm comm, const bool local)
{
  const int input = local ? 1 : 0;
  int output = 0;
  MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, comm);
  return output == 1;
}

bool FacePoint(mfem::Mesh& mesh,
               const mfem::FiniteElementCollection& collection,
               int face, int node, double& x, double& y);

double CanonicalOrientationValue(const int component, const double x,
                                 const double y)
{
  return 10000.0 * static_cast<double>(component + 1) +
      97.0 * x + 193.0 * y + 3.0 * x * y + 5.0 * y * y;
}

double ProjectedOrientationValue(const mfem::Vector& point,
                                 const int component)
{
  const double* const coordinates = point.HostRead();
  if (point.Size() != 2 || coordinates == nullptr)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return CanonicalOrientationValue(
      component, coordinates[0], coordinates[1]);
}

bool CheckAnalyticSegmentOrientations()
{
  for (int polynomial_order = 1; polynomial_order <= 4; ++polynomial_order)
  {
    mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
        2, 1, mfem::Element::QUADRILATERAL, false, 2.0, 1.0, false);
    mfem::ParMesh mesh(MPI_COMM_SELF, serial_mesh);
    mfem::DG_Interface_FECollection collection(
        polynomial_order, 2, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace space(
        &mesh, &collection, vector_dimension, mfem::Ordering::byNODES);
    DGInterfaceTraceMap map;
    if (!BuildDGInterfaceTraceMap(space, map) || !map.IsConsistent())
    {
      return false;
    }

    mfem::FunctionCoefficient component_zero(
        [](const mfem::Vector& point)
        { return ProjectedOrientationValue(point, 0); });
    mfem::FunctionCoefficient component_one(
        [](const mfem::Vector& point)
        { return ProjectedOrientationValue(point, 1); });
    mfem::FunctionCoefficient component_two(
        [](const mfem::Vector& point)
        { return ProjectedOrientationValue(point, 2); });
    mfem::FunctionCoefficient component_three(
        [](const mfem::Vector& point)
        { return ProjectedOrientationValue(point, 3); });
    mfem::Coefficient* coefficients[vector_dimension] = {
        &component_zero, &component_one, &component_two, &component_three};
    mfem::ParGridFunction projected(&space);
    projected = std::numeric_limits<double>::quiet_NaN();
    double* const projected_write = projected.HostWrite();
    const int face_dofs = polynomial_order + 1;
    if (projected_write == nullptr ||
        space.GetNDofs() != mesh.GetNumFaces() * face_dofs)
    {
      return false;
    }
    for (int face = 0; face < mesh.GetNumFaces(); ++face)
    {
      mfem::Array<int> edges;
      mfem::Array<int> orientations;
      mesh.GetFaceEdges(face, edges, orientations);
      const int* const edge_data = edges.HostRead();
      const int* const orientation_data = orientations.HostRead();
      const mfem::FiniteElement* const face_element =
          space.GetFaceElement(face);
      mfem::ElementTransformation* const transformation =
          mesh.GetFaceTransformation(face);
      if (edges.Size() != 1 || orientations.Size() != 1 ||
          edge_data == nullptr || orientation_data == nullptr ||
          edge_data[0] < 0 || edge_data[0] >= mesh.GetNEdges() ||
          (orientation_data[0] != 1 && orientation_data[0] != -1) ||
          face_element == nullptr || transformation == nullptr ||
          face_element->GetDof() != face_dofs)
      {
        return false;
      }
      mfem::Vector face_values(face_dofs);
      for (int component = 0; component < vector_dimension; ++component)
      {
        face_element->Project(
            *coefficients[component], *transformation, face_values);
        const double* const face_data = face_values.HostRead();
        if (face_data == nullptr) { return false; }
        for (int node = 0; node < face_dofs; ++node)
        {
          const int storage_node = orientation_data[0] > 0 ?
              node : polynomial_order - node;
          const int scalar_dof = edge_data[0] * face_dofs + storage_node;
          const int vdof = space.DofToVDof(scalar_dof, component);
          if (vdof < 0 || vdof >= projected.Size()) { return false; }
          projected_write[vdof] = face_data[node];
        }
      }
    }
    const double* const projected_data = projected.HostRead();
    if (projected_data == nullptr) { return false; }

    for (const DGInterfaceFaceMap& production_face : map.faces)
    {
      const int production_face_dofs = static_cast<int>(
          production_face.signed_local_vdofs.size() /
          static_cast<std::size_t>(vector_dimension));
      if (production_face_dofs != face_dofs) { return false; }
      for (int node = 0; node < production_face_dofs; ++node)
      {
        double x = 0.0;
        double y = 0.0;
        if (!FacePoint(mesh, collection, production_face.local_face,
                       node, x, y))
        {
          return false;
        }
        for (int component = 0; component < vector_dimension; ++component)
        {
          const int signed_vdof = production_face.signed_local_vdofs[
              static_cast<std::size_t>(
                  node + production_face_dofs * component)];
          const int vdof = DecodeSignedVDof(signed_vdof);
          if (vdof < 0 || vdof >= projected.Size()) { return false; }
          const double gathered =
              static_cast<double>(SignedVDofSign(signed_vdof)) *
              projected_data[vdof];
          const double expected =
              CanonicalOrientationValue(component, x, y);
          if (!NearlyEqual(gathered, expected))
          {
            std::cerr << "orientation gather mismatch p=" << polynomial_order
                      << " face=" << production_face.local_face
                      << " node=" << node << " component=" << component
                      << " signed_vdof=" << signed_vdof
                      << " gathered=" << gathered
                      << " expected=" << expected << '\n';
            return false;
          }
        }
      }
    }

    int interior_face = -1;
    for (int face = 0; face < mesh.GetNumFaces(); ++face)
    {
      if (mesh.FaceIsInterior(face))
      {
        if (interior_face >= 0) { return false; }
        interior_face = face;
      }
    }
    if (interior_face < 0) { return false; }

    int element_one = -1;
    int element_two = -1;
    int info_one = -1;
    int info_two = -1;
    mesh.GetFaceElements(interior_face, &element_one, &element_two);
    mesh.GetFaceInfos(interior_face, &info_one, &info_two);
    if (element_one < 0 || element_two < 0 || info_one < 0 || info_two < 0)
    {
      return false;
    }
    const int local_face_one = info_one / 64;
    const int local_face_two = info_two / 64;
    mfem::Array<int> element_one_vdofs;
    mfem::Array<int> element_two_vdofs;
    space.GetElementVDofs(element_one, element_one_vdofs);
    space.GetElementVDofs(element_two, element_two_vdofs);
    if (element_one_vdofs.Size() != element_two_vdofs.Size() ||
        element_one_vdofs.Size() % vector_dimension != 0)
    {
      return false;
    }
    const int scalar_element_dofs =
        element_one_vdofs.Size() / vector_dimension;
    if (scalar_element_dofs != 4 * face_dofs) { return false; }
    const int* const element_one_data = element_one_vdofs.HostRead();
    const int* const element_two_data = element_two_vdofs.HostRead();
    if (element_one_data == nullptr || element_two_data == nullptr)
    {
      return false;
    }

    bool observed_nonpalindrome = false;
    for (int component = 0; component < vector_dimension; ++component)
    {
      std::vector<double> side_one_values;
      std::vector<double> side_two_values;
      std::vector<double> expected_values;
      side_one_values.reserve(static_cast<std::size_t>(face_dofs));
      side_two_values.reserve(static_cast<std::size_t>(face_dofs));
      expected_values.reserve(static_cast<std::size_t>(face_dofs));
      for (int node = 0; node < face_dofs; ++node)
      {
        const int side_one_index = local_face_one * face_dofs + node +
            scalar_element_dofs * component;
        const int side_two_index = local_face_two * face_dofs + node +
            scalar_element_dofs * component;
        if (side_one_index < 0 ||
            side_one_index >= element_one_vdofs.Size() ||
            side_two_index < 0 ||
            side_two_index >= element_two_vdofs.Size())
        {
          return false;
        }
        const int side_one_signed = element_one_data[side_one_index];
        const int side_two_signed = element_two_data[side_two_index];
        const int side_one = DecodeSignedVDof(side_one_signed);
        const int side_two = DecodeSignedVDof(side_two_signed);
        if (side_one < 0 || side_one >= projected.Size() ||
            side_two < 0 || side_two >= projected.Size())
        {
          return false;
        }
        side_one_values.push_back(
            static_cast<double>(SignedVDofSign(side_one_signed)) *
            projected_data[side_one]);
        side_two_values.push_back(
            static_cast<double>(SignedVDofSign(side_two_signed)) *
            projected_data[side_two]);
        double x = 0.0;
        double y = 0.0;
        if (!FacePoint(mesh, collection, interior_face, node, x, y))
        {
          return false;
        }
        expected_values.push_back(
            CanonicalOrientationValue(component, x, y));
      }

      bool side_one_forward = true;
      bool side_one_reverse = true;
      bool side_two_forward = true;
      bool side_two_reverse = true;
      for (int node = 0; node < face_dofs; ++node)
      {
        const std::size_t forward = static_cast<std::size_t>(node);
        const std::size_t reverse = static_cast<std::size_t>(
            polynomial_order - node);
        side_one_forward = NearlyEqual(
            side_one_values[forward], expected_values[forward]) &&
            side_one_forward;
        side_one_reverse = NearlyEqual(
            side_one_values[forward], expected_values[reverse]) &&
            side_one_reverse;
        side_two_forward = NearlyEqual(
            side_two_values[forward], expected_values[forward]) &&
            side_two_forward;
        side_two_reverse = NearlyEqual(
            side_two_values[forward], expected_values[reverse]) &&
            side_two_reverse;
      }
      if (!((side_one_forward && side_two_reverse) ||
            (side_one_reverse && side_two_forward)))
      {
        std::cerr << "orientation side mismatch p=" << polynomial_order
                  << " component=" << component
                  << " side_one_forward=" << side_one_forward
                  << " side_one_reverse=" << side_one_reverse
                  << " side_two_forward=" << side_two_forward
                  << " side_two_reverse=" << side_two_reverse << '\n';
        return false;
      }
      observed_nonpalindrome = !NearlyEqual(
          expected_values.front(), expected_values.back()) ||
          observed_nonpalindrome;
    }
    if (!observed_nonpalindrome) { return false; }
  }
  return true;
}

bool CheckZeroOwnedMapContract()
{
  DGInterfaceTraceMap map;
  map.vector_dimension = 1;
  map.ordering = mfem::Ordering::byNODES;
  map.dofs.rank = 0;
  map.dofs.rank_count = 2;
  map.dofs.local_true_dof_count = 0;
  map.dofs.global_true_dof_count = 1;
  map.dofs.entries.push_back({0, 0, -1, 1, Ownership::ghost});
  map.faces.push_back({0, mfem::Geometry::SEGMENT, std::vector<int>{0}});
  return map.IsConsistent() && map.OwnedCount() == 0 &&
      map.GhostCount() == 1;
}

bool CheckUnsupported3D()
{
  mfem::Mesh valid_serial = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  mfem::ParMesh valid_mesh(MPI_COMM_SELF, valid_serial);
  mfem::DG_Interface_FECollection valid_collection(
      1, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace valid_space(
      &valid_mesh, &valid_collection, vector_dimension,
      mfem::Ordering::byNODES);
  DGInterfaceTraceExchangePlan preserved_plan;
  if (!BuildDGInterfaceTraceExchangePlan(valid_space, preserved_plan) ||
      !preserved_plan.IsConsistent())
  {
    return false;
  }
  const mfem::ParFiniteElementSpace* const preserved_space =
      preserved_plan.space;
  const long preserved_space_sequence = preserved_plan.space_sequence;
  const long preserved_mesh_sequence = preserved_plan.mesh_sequence;
  const DGInterfaceTraceMap preserved_plan_map = preserved_plan.map;

  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian3D(
      1, 1, 1, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial_mesh);
  mfem::DG_Interface_FECollection collection(
      1, 3, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, vector_dimension, mfem::Ordering::byNODES);
  DGInterfaceTraceMap map = SentinelMap();
  const DGInterfaceTraceMap map_before = map;
  return !BuildDGInterfaceTraceMap(space, map) &&
      SameTraceMap(map, map_before) &&
      !BuildDGInterfaceTraceExchangePlan(space, preserved_plan) &&
      preserved_plan.space == preserved_space &&
      preserved_plan.space_sequence == preserved_space_sequence &&
      preserved_plan.mesh_sequence == preserved_mesh_sequence &&
      SameTraceMap(preserved_plan.map, preserved_plan_map) &&
      preserved_plan.IsConsistent();
}

bool CheckFaceMaps(const mfem::ParFiniteElementSpace& space,
                   const DGInterfaceTraceMap& map)
{
  if (space.GetMesh() == nullptr || !map.IsConsistent() ||
      map.faces.size() !=
          static_cast<std::size_t>(space.GetMesh()->GetNumFaces()))
  {
    return false;
  }
  for (const auto& face : map.faces)
  {
    mfem::Array<int> direct;
    space.GetFaceVDofs(face.local_face, direct);
    if (face.geometry != mfem::Geometry::SEGMENT ||
        face.geometry != space.GetMesh()->GetFaceGeometry(face.local_face) ||
        direct.Size() != static_cast<int>(face.signed_local_vdofs.size()))
    {
      return false;
    }
    const int* const direct_data = direct.HostRead();
    for (int i = 0; i < direct.Size(); ++i)
    {
      const int signed_vdof = face.signed_local_vdofs[
          static_cast<std::size_t>(i)];
      const int vdof = DecodeSignedVDof(signed_vdof);
      if (signed_vdof != direct_data[i] || vdof < 0 ||
          vdof >= space.GetVSize())
      {
        return false;
      }
    }
  }
  return true;
}

bool FacePoint(mfem::Mesh& mesh,
               const mfem::FiniteElementCollection& collection,
               const int face, const int node, double& x, double& y)
{
  const mfem::FiniteElement* const face_element =
      collection.FiniteElementForGeometry(mfem::Geometry::SEGMENT);
  mfem::ElementTransformation* const transformation =
      mesh.GetFaceTransformation(face);
  if (face_element == nullptr || transformation == nullptr || node < 0 ||
      node >= face_element->GetNodes().GetNPoints())
  {
    return false;
  }
  mfem::Vector physical(mesh.SpaceDimension());
  transformation->Transform(face_element->GetNodes().IntPoint(node), physical);
  if (physical.Size() != 2) { return false; }
  const double* const data = physical.HostRead();
  x = data[0];
  y = data[1];
  return std::isfinite(x) && std::isfinite(y);
}

double GeometricTraceValue(const double x, const double y,
                           const int component)
{
  return 1000.0 * static_cast<double>(component + 1) +
      17.0 * x + 31.0 * y + 0.5 * x * y;
}

double GeometricAction(const double value, const double x, const double y,
                       const int component)
{
  return value * (1.0 + 0.25 * static_cast<double>(component + 1) +
                  0.5 * x + 0.75 * y);
}

bool BuildExpectedLocalValues(mfem::ParMesh& mesh,
                              const mfem::FiniteElementCollection& collection,
                              const DGInterfaceTraceMap& map,
                              mfem::Vector& expected)
{
  expected.SetSize(static_cast<int>(map.dofs.entries.size()));
  double* const expected_data = expected.HostWrite();
  std::vector<int> seen(map.dofs.entries.size(), 0);
  for (const auto& face : map.faces)
  {
    const int scalar_face_dofs = static_cast<int>(
        face.signed_local_vdofs.size() /
        static_cast<std::size_t>(map.vector_dimension));
    for (int node = 0; node < scalar_face_dofs; ++node)
    {
      double x = 0.0;
      double y = 0.0;
      if (!FacePoint(mesh, collection, face.local_face, node, x, y))
      {
        return false;
      }
      for (int component = 0; component < map.vector_dimension; ++component)
      {
        const int signed_vdof = face.signed_local_vdofs[
            static_cast<std::size_t>(node + scalar_face_dofs * component)];
        const int vdof = DecodeSignedVDof(signed_vdof);
        if (vdof < 0 || vdof >= expected.Size() ||
            seen[static_cast<std::size_t>(vdof)] != 0)
        {
          return false;
        }
        seen[static_cast<std::size_t>(vdof)] = 1;
        expected_data[vdof] =
            static_cast<double>(SignedVDofSign(signed_vdof)) *
            GeometricTraceValue(x, y, component);
      }
    }
  }
  return std::find(seen.begin(), seen.end(), 0) == seen.end();
}

bool CheckFailureAtomicity(mfem::ParMesh& mesh,
                           mfem::DG_Interface_FECollection& collection,
                           mfem::ParFiniteElementSpace& space,
                           DGInterfaceTraceExchangePlan& plan,
                           const mfem::Vector& true_values,
                           const mfem::Vector& distributed)
{
  bool passed = true;

  mfem::H1_FECollection wrong_collection(1, 2);
  mfem::ParFiniteElementSpace wrong_space(
      &mesh, &wrong_collection, vector_dimension, mfem::Ordering::byNODES);
  DGInterfaceTraceMap preserved_map = SentinelMap();
  const DGInterfaceTraceMap preserved_map_before = preserved_map;
  passed = !BuildDGInterfaceTraceMap(wrong_space, preserved_map) &&
      SameTraceMap(preserved_map, preserved_map_before) && passed;

  mfem::ParFiniteElementSpace wrong_order(
      &mesh, &collection, vector_dimension, mfem::Ordering::byVDIM);
  preserved_map = SentinelMap();
  const DGInterfaceTraceMap wrong_order_before = preserved_map;
  passed = !BuildDGInterfaceTraceMap(wrong_order, preserved_map) &&
      SameTraceMap(preserved_map, wrong_order_before) && passed;

  const mfem::ParFiniteElementSpace* const preserved_space = plan.space;
  const long preserved_space_sequence = plan.space_sequence;
  const long preserved_mesh_sequence = plan.mesh_sequence;
  const DGInterfaceTraceMap preserved_plan_map = plan.map;
  passed = !BuildDGInterfaceTraceExchangePlan(wrong_space, plan) &&
      plan.space == preserved_space &&
      plan.space_sequence == preserved_space_sequence &&
      plan.mesh_sequence == preserved_mesh_sequence &&
      SameTraceMap(plan.map, preserved_plan_map) && plan.IsConsistent() &&
      passed;
  passed = !BuildDGInterfaceTraceExchangePlan(wrong_order, plan) &&
      plan.space == preserved_space &&
      plan.space_sequence == preserved_space_sequence &&
      plan.mesh_sequence == preserved_mesh_sequence &&
      SameTraceMap(plan.map, preserved_plan_map) && plan.IsConsistent() &&
      passed;

  const int original_dimension = plan.map.vector_dimension;
  plan.map.vector_dimension = 0;
  mfem::Vector wrong_dimension_output(space.GetVSize());
  wrong_dimension_output = -17.0;
  const mfem::Vector wrong_dimension_before(wrong_dimension_output);
  const bool wrong_dimension_rejected = !plan.IsConsistent() &&
      !plan.Distribute(space, true_values, wrong_dimension_output) &&
      SameValues(wrong_dimension_output, wrong_dimension_before);
  plan.map.vector_dimension = original_dimension;
  passed = wrong_dimension_rejected && plan.IsConsistent() && passed;

  mfem::Vector non_finite(true_values);
  if (non_finite.Size() > 0)
  {
    non_finite.HostWrite()[0] = std::numeric_limits<double>::quiet_NaN();
  }
  mfem::Vector preserved_output(space.GetVSize());
  preserved_output = -19.0;
  const mfem::Vector preserved_output_before(preserved_output);
  passed = !plan.Distribute(space, non_finite, preserved_output) &&
      SameValues(preserved_output, preserved_output_before) && passed;

  mfem::Vector empty_output;
  passed = !plan.Distribute(space, true_values, empty_output) &&
      empty_output.Size() == 0 && passed;

  const int backing_size = std::max(space.GetTrueVSize(), space.GetVSize()) + 2;
  mfem::Vector overlap_backing(backing_size);
  double* const overlap_data = overlap_backing.HostWrite();
  for (int i = 0; i < overlap_backing.Size(); ++i)
  {
    overlap_data[i] = 200.0 + static_cast<double>(i);
  }
  mfem::Vector overlap_input;
  mfem::Vector overlap_output;
  overlap_input.MakeRef(overlap_backing, 0, space.GetTrueVSize());
  overlap_output.MakeRef(overlap_backing, 1, space.GetVSize());
  const mfem::Vector overlap_before(overlap_backing);
  passed = !plan.Distribute(space, overlap_input, overlap_output) &&
      SameValues(overlap_backing, overlap_before) && passed;

  mfem::Vector view_backing(space.GetVSize() + 2);
  view_backing = -23.0;
  mfem::Vector view;
  view.MakeRef(view_backing, 1, space.GetVSize());
  const double* const view_data = view.HostRead();
  passed = plan.Distribute(space, true_values, view) && !view.OwnsData() &&
      view.HostRead() == view_data && SameValues(view, distributed) && passed;

  mfem::Vector wrong_view_backing(space.GetVSize() + 1);
  wrong_view_backing = -29.0;
  mfem::Vector wrong_view;
  wrong_view.MakeRef(wrong_view_backing, 0, space.GetVSize() - 1);
  const mfem::Vector wrong_view_before(wrong_view_backing);
  passed = !plan.Distribute(space, true_values, wrong_view) &&
      SameValues(wrong_view_backing, wrong_view_before) && passed;

  mfem::ParFiniteElementSpace unrelated(
      &mesh, &collection, vector_dimension, mfem::Ordering::byNODES);
  mfem::Vector unrelated_true(unrelated.GetTrueVSize());
  unrelated_true = 1.0;
  mfem::Vector unrelated_output(space.GetVSize());
  unrelated_output = -31.0;
  const mfem::Vector unrelated_before(unrelated_output);
  passed = !plan.Distribute(unrelated, unrelated_true, unrelated_output) &&
      SameValues(unrelated_output, unrelated_before) && passed;

  if (plan.map.dofs.entries.size() > 1)
  {
    std::swap(plan.map.dofs.entries[0].global_true_dof,
              plan.map.dofs.entries[1].global_true_dof);
    mfem::Vector altered_output(space.GetVSize());
    altered_output = -37.0;
    const mfem::Vector altered_before(altered_output);
    const bool altered_rejected =
        !plan.Distribute(space, true_values, altered_output) &&
        SameValues(altered_output, altered_before);
    std::swap(plan.map.dofs.entries[0].global_true_dof,
              plan.map.dofs.entries[1].global_true_dof);
    passed = altered_rejected && plan.IsConsistent() && passed;
  }

  return passed;
}

bool CheckAsymmetricFailures(MPI_Comm comm, const int rank,
                             mfem::ParFiniteElementSpace& space,
                             const DGInterfaceTraceExchangePlan& plan,
                             const mfem::Vector& true_values)
{
  mfem::Vector asymmetric_true(true_values);
  if (rank == 0 && asymmetric_true.Size() > 0)
  {
    asymmetric_true.HostWrite()[0] =
        std::numeric_limits<double>::quiet_NaN();
  }
  mfem::Vector finite_output(space.GetVSize());
  finite_output = -43.0 - static_cast<double>(rank);
  const mfem::Vector finite_before(finite_output);
  bool local = !plan.Distribute(space, asymmetric_true, finite_output) &&
      SameValues(finite_output, finite_before);
  if (!GlobalAll(comm, local)) { return false; }

  if (space.GetTrueVSize() <= 0) { return false; }
  mfem::Vector null_true(
      static_cast<double*>(nullptr), space.GetTrueVSize());
  const mfem::Vector& supplied_true = rank == 0 ? null_true : true_values;
  mfem::Vector null_output(space.GetVSize());
  null_output = -45.0 - static_cast<double>(rank);
  const mfem::Vector null_before(null_output);
  local = !plan.Distribute(space, supplied_true, null_output) &&
      SameValues(null_output, null_before);
  if (!GlobalAll(comm, local)) { return false; }

  mfem::Mesh self_serial_mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  mfem::ParMesh self_mesh(MPI_COMM_SELF, self_serial_mesh);
  mfem::DG_Interface_FECollection self_collection(
      1, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace self_space(
      &self_mesh, &self_collection, vector_dimension, mfem::Ordering::byNODES);
  mfem::ParFiniteElementSpace& supplied_space =
      rank == 0 ? self_space : space;
  mfem::Vector communicator_output(space.GetVSize());
  communicator_output = -46.0 - static_cast<double>(rank);
  const mfem::Vector communicator_before(communicator_output);
  local = !plan.Distribute(
      supplied_space, true_values, communicator_output) &&
      SameValues(communicator_output, communicator_before);
  if (!GlobalAll(comm, local)) { return false; }

  mfem::Vector shape_output(
      rank == 0 ? space.GetVSize() - 1 : space.GetVSize());
  shape_output = -47.0 - static_cast<double>(rank);
  const mfem::Vector shape_before(shape_output);
  local = !plan.Distribute(space, true_values, shape_output) &&
      SameValues(shape_output, shape_before);
  return GlobalAll(comm, local);
}

std::vector<int> BuildPartition(const mfem::Mesh& mesh, const int rank_count)
{
  std::vector<int> partition(static_cast<std::size_t>(mesh.GetNE()), 0);
  if (rank_count == 1) { return partition; }
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    mfem::Array<int> vertices;
    mesh.GetElementVertices(element, vertices);
    const int* const vertex_data = vertices.HostRead();
    double center_x = 0.0;
    for (int i = 0; i < vertices.Size(); ++i)
    {
      center_x += mesh.GetVertex(vertex_data[i])[0];
    }
    center_x /= static_cast<double>(vertices.Size());
    partition[static_cast<std::size_t>(element)] = center_x < 0.5 ? 0 : 1;
  }
  return partition;
}

bool CheckZeroLocalMapPlanAndExchange(MPI_Comm comm)
{
  int rank = -1;
  int rank_count = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &rank_count);
  if (rank_count == 1) { return true; }

  // One globally owned cell leaves rank one with no local elements, faces, or
  // trace dofs.  It must nevertheless retain an empty plan and participate in
  // the same plan distribution as the owner rank.
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  const std::vector<int> partition(1, 0);
  mfem::ParMesh mesh(comm, serial_mesh, partition.data());
  mfem::DG_Interface_FECollection collection(
      1, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, vector_dimension, mfem::Ordering::byNODES);

  DGInterfaceTraceMap map;
  DGInterfaceTraceExchangePlan plan;
  bool local = BuildDGInterfaceTraceMap(space, map) && map.IsConsistent() &&
      BuildDGInterfaceTraceExchangePlan(space, plan) && plan.IsConsistent();
  const bool empty_rank = mesh.GetNE() == 0;
  local = local && (empty_rank
      ? (space.GetVSize() == 0 && space.GetTrueVSize() == 0 &&
         mesh.GetNumFaces() == 0 && map.dofs.entries.empty() &&
         map.faces.empty() && plan.map.dofs.entries.empty() &&
         plan.map.faces.empty())
      : (space.GetVSize() > 0 && space.GetTrueVSize() > 0 &&
         !map.dofs.entries.empty() && !map.faces.empty()));
  if (!GlobalAll(comm, local)) { return false; }

  mfem::Vector true_values(space.GetTrueVSize());
  true_values = 1.0;
  mfem::Vector distributed(space.GetVSize());
  distributed = -1.0;
  local = plan.Distribute(space, true_values, distributed);
  const double* const distributed_data = distributed.HostRead();
  local = local && (distributed_data != nullptr || distributed.Size() == 0);
  for (int dof = 0; dof < distributed.Size(); ++dof)
  {
    local = NearlyEqual(distributed_data[dof], 1.0) && local;
  }
  return GlobalAll(comm, local);
}

double SerialActionOracle(mfem::Mesh& mesh,
                          const mfem::FiniteElementCollection& collection)
{
  const int face_dofs =
      collection.DofForGeometry(mfem::Geometry::SEGMENT);
  double action = 0.0;
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
    for (int node = 0; node < face_dofs; ++node)
    {
      double x = 0.0;
      double y = 0.0;
      if (!FacePoint(mesh, collection, face, node, x, y))
      {
        return std::numeric_limits<double>::quiet_NaN();
      }
      for (int component = 0; component < vector_dimension; ++component)
      {
        const double value = GeometricTraceValue(x, y, component);
        action += GeometricAction(value, x, y, component);
      }
    }
  return action;
}

bool CheckMapExchangeAndParity(MPI_Comm comm)
{
  int rank = -1;
  int rank_count = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &rank_count);
  // This parity partition is intentionally two-way; the zero-local plan test
  // below covers all larger communicator sizes with its rank-zero-owned cell.
  if (rank_count > 2) { return true; }

  constexpr int elements_x = 4;
  constexpr int elements_y = 2;
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      elements_x, elements_y, mfem::Element::QUADRILATERAL,
      false, 1.0, 1.0, false);
  const int global_faces = serial_mesh.GetNumFaces();
  const std::vector<int> partition = BuildPartition(serial_mesh, rank_count);
  mfem::ParMesh parallel_mesh(comm, serial_mesh, partition.data());
  mfem::DG_Interface_FECollection collection(
      1, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &parallel_mesh, &collection, vector_dimension, mfem::Ordering::byNODES);

  int rank_zero_plan_ok = 1;
  if (rank == 0)
  {
    DGInterfaceTraceExchangePlan rank_zero_plan;
    rank_zero_plan_ok =
        BuildDGInterfaceTraceExchangePlan(trace_space, rank_zero_plan) &&
        rank_zero_plan.IsConsistent() ? 1 : 0;
  }
  MPI_Bcast(&rank_zero_plan_ok, 1, MPI_INT, 0, comm);
  if (rank_zero_plan_ok != 1) { return false; }

  DGInterfaceTraceExchangePlan plan;
  const bool local_plan_ready =
      BuildDGInterfaceTraceExchangePlan(trace_space, plan) &&
      plan.IsConsistent() && CheckFaceMaps(trace_space, plan.map);
  if (!GlobalAll(comm, local_plan_ready)) { return false; }

  const GlobalIndex expected_global_dofs = static_cast<GlobalIndex>(
      global_faces * 2 * vector_dimension);
  bool local_map_evidence =
      plan.map.vector_dimension == vector_dimension &&
      plan.map.ordering == mfem::Ordering::byNODES &&
      plan.map.dofs.global_true_dof_count == expected_global_dofs &&
      plan.map.dofs.entries.size() ==
          static_cast<std::size_t>(trace_space.GetVSize()) &&
      plan.map.OwnedCount() ==
          static_cast<std::size_t>(trace_space.GetTrueVSize());

  std::vector<int> local_owner_count(
      static_cast<std::size_t>(expected_global_dofs), 0);
  std::vector<int> actual_ghost_local_dofs;
  for (const TraceMapEntry& entry : plan.map.dofs.entries)
  {
    if (entry.global_true_dof < 0 ||
        entry.global_true_dof >= expected_global_dofs)
    {
      local_map_evidence = false;
      continue;
    }
    if (entry.ownership == Ownership::owned)
    {
      ++local_owner_count[static_cast<std::size_t>(entry.global_true_dof)];
    }
    else
    {
      actual_ghost_local_dofs.push_back(entry.local_dof);
    }
  }
  std::vector<int> global_owner_count(local_owner_count.size(), 0);
  MPI_Allreduce(local_owner_count.data(), global_owner_count.data(),
                static_cast<int>(local_owner_count.size()), MPI_INT, MPI_SUM,
                comm);
  local_map_evidence = local_map_evidence &&
      std::find_if(global_owner_count.begin(), global_owner_count.end(),
                   [](const int count) { return count != 1; }) ==
          global_owner_count.end();

  std::vector<int> expected_ghost_local_dofs;
  std::vector<int> shared_local_faces;
  const int expected_shared_faces = rank_count == 2 ? elements_y : 0;
  local_map_evidence = local_map_evidence &&
      parallel_mesh.GetNSharedFaces() == expected_shared_faces;
  for (int shared = 0; shared < parallel_mesh.GetNSharedFaces(); ++shared)
  {
    const int local_face = parallel_mesh.GetSharedFace(shared);
    shared_local_faces.push_back(local_face);
    if (rank == 1)
    {
      mfem::Array<int> direct;
      trace_space.GetFaceVDofs(local_face, direct);
      const int* const direct_data = direct.HostRead();
      for (int i = 0; i < direct.Size(); ++i)
      {
        expected_ghost_local_dofs.push_back(
            DecodeSignedVDof(direct_data[i]));
      }
    }
  }
  std::sort(expected_ghost_local_dofs.begin(),
            expected_ghost_local_dofs.end());
  std::sort(actual_ghost_local_dofs.begin(), actual_ghost_local_dofs.end());
  local_map_evidence = local_map_evidence &&
      actual_ghost_local_dofs == expected_ghost_local_dofs;
  const unsigned long long expected_global_ghosts = rank_count == 2 ?
      static_cast<unsigned long long>(
          elements_y * 2 * vector_dimension) : 0ULL;
  const unsigned long long local_ghosts =
      static_cast<unsigned long long>(plan.map.GhostCount());
  unsigned long long global_ghosts = 0;
  MPI_Allreduce(&local_ghosts, &global_ghosts, 1, MPI_UNSIGNED_LONG_LONG,
                MPI_SUM, comm);
  local_map_evidence = local_map_evidence &&
      global_ghosts == expected_global_ghosts;
  if (!GlobalAll(comm, local_map_evidence)) { return false; }

  mfem::Vector expected_local;
  bool local_value_evidence = BuildExpectedLocalValues(
      parallel_mesh, collection, plan.map, expected_local);
  mfem::Vector true_values(trace_space.GetTrueVSize());
  true_values = std::numeric_limits<double>::quiet_NaN();
  const double* const expected_data = expected_local.HostRead();
  double* const true_data = true_values.HostWrite();
  for (const TraceMapEntry& entry : plan.map.dofs.entries)
  {
    if (entry.ownership == Ownership::owned)
    {
      true_data[entry.local_true_dof] = expected_data[entry.local_dof];
    }
  }
  const double* const finite_true_data = true_values.HostRead();
  for (int i = 0; i < true_values.Size(); ++i)
  {
    local_value_evidence =
        std::isfinite(finite_true_data[i]) && local_value_evidence;
  }
  if (!GlobalAll(comm, local_value_evidence)) { return false; }

  mfem::Vector distributed(trace_space.GetVSize());
  const bool local_distributed = plan.Distribute(
      trace_space, true_values, distributed);
  if (!GlobalAll(comm, local_distributed)) { return false; }
  const double* const distributed_data = distributed.HostRead();
  for (const TraceMapEntry& entry : plan.map.dofs.entries)
  {
    local_value_evidence = NearlyEqual(
        distributed_data[entry.local_dof], expected_data[entry.local_dof]) &&
        local_value_evidence;
  }

  double local_owned_action = 0.0;
  double local_shared_action = 0.0;
  double expected_shared_action = 0.0;
  for (const auto& face : plan.map.faces)
  {
    const int scalar_face_dofs = static_cast<int>(
        face.signed_local_vdofs.size() /
        static_cast<std::size_t>(vector_dimension));
    const bool shared_face = std::find(
        shared_local_faces.begin(), shared_local_faces.end(),
        face.local_face) != shared_local_faces.end();
    for (int node = 0; node < scalar_face_dofs; ++node)
    {
      double x = 0.0;
      double y = 0.0;
      if (!FacePoint(parallel_mesh, collection, face.local_face, node, x, y))
      {
        local_value_evidence = false;
        continue;
      }
      for (int component = 0; component < vector_dimension; ++component)
      {
        const int signed_vdof = face.signed_local_vdofs[
            static_cast<std::size_t>(node + scalar_face_dofs * component)];
        const int vdof = DecodeSignedVDof(signed_vdof);
        const double logical_value =
            static_cast<double>(SignedVDofSign(signed_vdof)) *
            distributed_data[vdof];
        const double expected_value = GeometricTraceValue(x, y, component);
        local_value_evidence = NearlyEqual(logical_value, expected_value) &&
            local_value_evidence;
        if (plan.map.dofs.entries[static_cast<std::size_t>(vdof)].ownership ==
            Ownership::owned)
        {
          local_owned_action +=
              GeometricAction(logical_value, x, y, component);
        }
        if (shared_face)
        {
          local_shared_action +=
              GeometricAction(logical_value, x, y, component);
          expected_shared_action +=
              GeometricAction(expected_value, x, y, component);
        }
      }
    }
  }
  local_value_evidence = NearlyEqual(
      local_shared_action, expected_shared_action) && local_value_evidence;
  double global_owned_action = 0.0;
  MPI_Allreduce(&local_owned_action, &global_owned_action, 1, MPI_DOUBLE,
                MPI_SUM, comm);
  const double serial_action = SerialActionOracle(serial_mesh, collection);
  local_value_evidence = std::isfinite(serial_action) &&
      NearlyEqual(global_owned_action, serial_action) &&
      local_value_evidence;

  const int local_evidence_rank = local_value_evidence ? 1 : 0;
  int evidence_ranks = 0;
  MPI_Allreduce(&local_evidence_rank, &evidence_ranks, 1, MPI_INT, MPI_SUM,
                comm);
  if (evidence_ranks != rank_count) { return false; }

  if (!GlobalAll(comm, CheckFailureAtomicity(
          parallel_mesh, collection, trace_space, plan, true_values,
          distributed)))
  {
    return false;
  }
  if (!CheckAsymmetricFailures(comm, rank, trace_space, plan, true_values))
  {
    return false;
  }

  mfem::Vector stale_output(trace_space.GetVSize());
  stale_output = -41.0;
  const mfem::Vector stale_before(stale_output);
  parallel_mesh.UniformRefinement();
  bool local_stale = !plan.IsConsistent() &&
      !plan.Distribute(trace_space, true_values, stale_output) &&
      SameValues(stale_output, stale_before);
  trace_space.Update(false);
  local_stale = local_stale && !plan.IsConsistent() &&
      !plan.Distribute(trace_space, true_values, stale_output) &&
      SameValues(stale_output, stale_before);
  return GlobalAll(comm, local_stale);
}

void RecordCheck(const std::string& name, const bool passed, int& failures,
                 const int rank)
{
  if (!passed)
  {
    ++failures;
    std::cerr << "FAIL rank=" << rank << " check=" << name << '\n';
  }
}

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int rank = -1;
  int rank_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
  if (rank_count < 1)
  {
    MPI_Finalize();
    return 75;
  }

  int failures = 0;
  RecordCheck("analytic_segment_orientations_p1_p4",
              CheckAnalyticSegmentOrientations(), failures, rank);
  RecordCheck("zero_owned_true_dof_map_contract",
              CheckZeroOwnedMapContract(), failures, rank);
  RecordCheck("zero_local_dg_trace_map_plan_exchange",
              CheckZeroLocalMapPlanAndExchange(MPI_COMM_WORLD), failures,
              rank);
  RecordCheck("unsupported_3d_rejected", CheckUnsupported3D(), failures, rank);
  RecordCheck("dg_interface_map_exchange_and_parity",
              CheckMapExchangeAndParity(MPI_COMM_WORLD), failures, rank);

  const int local_passed = failures == 0 ? 1 : 0;
  int global_passed = 0;
  MPI_Allreduce(&local_passed, &global_passed, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Finalize();
  return global_passed == 1 ? 0 : 1;
}

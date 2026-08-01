#include "sc25_scalar_igr_independent_oracle.hpp"

#include "numerics/hdg/runtime/sha256.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dg_igr_hdg_sc25_cpu_test
{
namespace
{

namespace sc25 = dg::hdg::runtime::cpu::sc25;

constexpr double assembled_action_tolerance = 2.0e-11;
constexpr double frozen_gamma = 1.4;

struct IndependentElementRestriction final
{
  int element = -1;
  int scalar_volume_dofs = 0;
  int scalar_face_dofs = 0;
  int local_interior_unknowns = 0;
  int local_trace_unknowns = 0;
  std::vector<int> faces;
  std::vector<int> orientations;
  std::vector<IndependentRestrictionEntry> entries;
};

void Require(const bool condition, const std::string& message,
             IndependentC0Evidence& evidence)
{
  if (!condition) { evidence.failures.push_back(message); }
}

double ExactTolerance(const double expected_norm_inf)
{
  return 64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, expected_norm_inf);
}

template <std::size_t N>
double NormInf(const std::array<double, N>& values)
{
  double result = 0.0;
  for (const double value : values)
  {
    result = std::max(result, std::abs(value));
  }
  return result;
}

double NormInf(const std::vector<double>& values)
{
  double result = 0.0;
  for (const double value : values)
  {
    result = std::max(result, std::abs(value));
  }
  return result;
}

double RelativeL2(const double* actual, const double* expected,
                  const int size)
{
  long double difference_square = 0.0L;
  long double expected_square = 0.0L;
  for (int entry = 0; entry < size; ++entry)
  {
    const long double difference =
        static_cast<long double>(actual[entry]) - expected[entry];
    difference_square += difference * difference;
    expected_square += static_cast<long double>(expected[entry]) *
        expected[entry];
  }
  return std::sqrt(static_cast<double>(difference_square)) /
      std::max(1.0, std::sqrt(static_cast<double>(expected_square)));
}

bool DirectAugmentedPartialPivotSolve(
    const std::vector<double>& original_matrix, const int size,
    const std::vector<double>& rhs, std::vector<double>& solution)
{
  if (size <= 0 ||
      original_matrix.size() != static_cast<std::size_t>(size) * size ||
      rhs.size() != static_cast<std::size_t>(size))
  {
    return false;
  }
  std::vector<double> matrix = original_matrix;
  solution = rhs;
  for (const double value : matrix)
  {
    if (!std::isfinite(value)) { return false; }
  }
  for (const double value : solution)
  {
    if (!std::isfinite(value)) { return false; }
  }
  for (int column = 0; column < size; ++column)
  {
    int selected = column;
    double maximum = std::abs(matrix[
        static_cast<std::size_t>(column) * size + column]);
    for (int row = column + 1; row < size; ++row)
    {
      const double candidate = std::abs(matrix[
          static_cast<std::size_t>(row) * size + column]);
      if (candidate > maximum)
      {
        maximum = candidate;
        selected = row;
      }
    }
    if (!std::isfinite(maximum) || !(maximum > 0.0)) { return false; }
    if (selected != column)
    {
      for (int entry = 0; entry < size; ++entry)
      {
        std::swap(matrix[static_cast<std::size_t>(selected) * size + entry],
                  matrix[static_cast<std::size_t>(column) * size + entry]);
      }
      std::swap(solution[selected], solution[column]);
    }
    const double diagonal = matrix[
        static_cast<std::size_t>(column) * size + column];
    for (int row = column + 1; row < size; ++row)
    {
      const double multiplier = matrix[
          static_cast<std::size_t>(row) * size + column] / diagonal;
      matrix[static_cast<std::size_t>(row) * size + column] = 0.0;
      for (int entry = column + 1; entry < size; ++entry)
      {
        matrix[static_cast<std::size_t>(row) * size + entry] -=
            multiplier * matrix[
                static_cast<std::size_t>(column) * size + entry];
      }
      solution[row] -= multiplier * solution[column];
    }
  }
  for (int row = size - 1; row >= 0; --row)
  {
    for (int column = row + 1; column < size; ++column)
    {
      solution[row] -= matrix[
          static_cast<std::size_t>(row) * size + column] *
          solution[column];
    }
    const double diagonal = matrix[
        static_cast<std::size_t>(row) * size + row];
    if (!std::isfinite(diagonal) || diagonal == 0.0) { return false; }
    solution[row] /= diagonal;
    if (!std::isfinite(solution[row])) { return false; }
  }
  return true;
}

bool RejectedInterleavedPartialPivotSolve3(
    const std::array<double, 9>& original_matrix,
    const std::array<double, 3>& rhs, std::array<int, 3>& pivots,
    std::array<double, 3>& solution)
{
  std::array<double, 9> matrix = original_matrix;
  for (int column = 0; column < 3; ++column)
  {
    int selected = column;
    for (int row = column + 1; row < 3; ++row)
    {
      if (std::abs(matrix[3 * row + column]) >
          std::abs(matrix[3 * selected + column]))
      {
        selected = row;
      }
    }
    pivots[column] = selected;
    if (matrix[3 * selected + column] == 0.0) { return false; }
    if (selected != column)
    {
      for (int entry = 0; entry < 3; ++entry)
      {
        std::swap(matrix[3 * selected + entry],
                  matrix[3 * column + entry]);
      }
    }
    for (int row = column + 1; row < 3; ++row)
    {
      matrix[3 * row + column] /= matrix[3 * column + column];
      for (int entry = column + 1; entry < 3; ++entry)
      {
        matrix[3 * row + entry] -=
            matrix[3 * row + column] * matrix[3 * column + entry];
      }
    }
  }

  solution = rhs;
  for (int column = 0; column < 3; ++column)
  {
    if (pivots[column] != column)
    {
      std::swap(solution[column], solution[pivots[column]]);
    }
    for (int row = column + 1; row < 3; ++row)
    {
      solution[row] -= matrix[3 * row + column] * solution[column];
    }
  }
  for (int row = 2; row >= 0; --row)
  {
    for (int column = row + 1; column < 3; ++column)
    {
      solution[row] -= matrix[3 * row + column] * solution[column];
    }
    solution[row] /= matrix[3 * row + row];
  }
  return true;
}

template <std::size_t N>
void RequireExactArray(const std::array<double, N>& actual,
                       const std::array<double, N>& expected,
                       const std::string& message,
                       IndependentC0Evidence& evidence)
{
  const double tolerance = ExactTolerance(NormInf(expected));
  for (std::size_t entry = 0; entry < N; ++entry)
  {
    Require(std::abs(actual[entry] - expected[entry]) <= tolerance,
            message + " entry " + std::to_string(entry), evidence);
  }
}

void RequireExactVector(const std::vector<double>& actual,
                        const std::vector<double>& expected,
                        const std::string& message,
                        IndependentC0Evidence& evidence)
{
  if (actual.size() != expected.size())
  {
    Require(false, message + " extent", evidence);
    return;
  }
  const double tolerance = ExactTolerance(NormInf(expected));
  for (std::size_t entry = 0; entry < expected.size(); ++entry)
  {
    Require(std::abs(actual[entry] - expected[entry]) <= tolerance,
            message + " entry " + std::to_string(entry), evidence);
  }
}

bool DecodeSignedDof(const int signed_dof, int& decoded, int& sign)
{
  if (signed_dof == std::numeric_limits<int>::min())
  {
    decoded = -1;
    sign = 0;
    return false;
  }
  sign = signed_dof < 0 ? -1 : 1;
  decoded = signed_dof < 0 ? -1 - signed_dof : signed_dof;
  return decoded >= 0;
}

void CheckSpace(const sc25::SpaceIdentity& actual,
                const mfem::ParMesh* mesh,
                const mfem::FiniteElementCollection* collection,
                const int vector_dimension, const char* name,
                IndependentC0Evidence& evidence)
{
  Require(actual.address != nullptr, std::string(name) + " address",
          evidence);
  Require(actual.mesh == mesh, std::string(name) + " mesh identity",
          evidence);
  Require(actual.collection == collection,
          std::string(name) + " collection identity", evidence);
  Require(actual.vector_dimension == vector_dimension,
          std::string(name) + " vector dimension", evidence);
  Require(actual.ordering == mfem::Ordering::byNODES,
          std::string(name) + " byNODES ordering", evidence);
  if (actual.address == nullptr) { return; }
  Require(actual.local_vector_size == actual.address->GetVSize(),
          std::string(name) + " local vector ownership", evidence);
  Require(actual.local_true_size == actual.address->GetTrueVSize(),
          std::string(name) + " local true ownership", evidence);
  Require(actual.global_true_size == actual.address->GlobalTrueVSize(),
          std::string(name) + " global true ownership", evidence);
  Require(actual.true_offset_begin == actual.address->GetMyTDofOffset() &&
              actual.true_offset_end ==
                  actual.true_offset_begin + actual.local_true_size,
          std::string(name) + " true interval", evidence);
}

bool BuildIndependentRestriction(
    const int element, const int degree, const mfem::ParMesh& mesh,
    const mfem::DG_Interface_FECollection& collection,
    const mfem::ParFiniteElementSpace& carrier_space,
    IndependentElementRestriction& output,
    IndependentC0Evidence& evidence)
{
  output = {};
  output.element = element;
  if (element < 0 || element >= mesh.GetNE() || degree < 1 ||
      carrier_space.GetVDim() != sc25::trace_components ||
      carrier_space.GetOrdering() != mfem::Ordering::byNODES ||
      carrier_space.GetVSize() <= 0 || carrier_space.GetTrueVSize() <= 0)
  {
    Require(false, "independent restriction precondition", evidence);
    return false;
  }
  output.scalar_volume_dofs = (degree + 1) * (degree + 1);
  output.scalar_face_dofs = degree + 1;
  output.local_interior_unknowns =
      sc25::local_components * output.scalar_volume_dofs;
  output.local_trace_unknowns = sc25::trace_components *
      sc25::faces_per_quadrilateral * output.scalar_face_dofs;

  mfem::Array<int> faces, orientations;
  mesh.GetElementEdges(element, faces, orientations);
  if (faces.Size() != sc25::faces_per_quadrilateral ||
      orientations.Size() != faces.Size())
  {
    Require(false, "independent element edge extent", evidence);
    return false;
  }
  output.faces.assign(faces.begin(), faces.end());
  output.orientations.assign(orientations.begin(), orientations.end());
  output.entries.resize(
      static_cast<std::size_t>(output.local_trace_unknowns));

  for (int local_face = 0; local_face < faces.Size(); ++local_face)
  {
    const int face = faces[local_face];
    if (face < 0 || face >= mesh.GetNumFaces() ||
        mesh.GetFaceGeometry(face) != mfem::Geometry::SEGMENT)
    {
      Require(false, "independent face identity", evidence);
      return false;
    }
    const mfem::FiniteElement* face_element =
        carrier_space.GetFaceElement(face);
    mfem::Array<int> face_vdofs;
    carrier_space.GetFaceVDofs(face, face_vdofs);
    if (face_element == nullptr ||
        face_element->GetDof() != output.scalar_face_dofs ||
        face_vdofs.Size() !=
            sc25::trace_components * output.scalar_face_dofs)
    {
      Require(false, "independent face VDof extent", evidence);
      return false;
    }
    const int* permutation = collection.DofOrderForOrientation(
        mfem::Geometry::SEGMENT, orientations[local_face]);
    if (permutation == nullptr)
    {
      Require(false, "independent orientation permutation", evidence);
      return false;
    }
    for (int component = 0; component < sc25::trace_components; ++component)
      for (int oriented = 0; oriented < output.scalar_face_dofs; ++oriented)
      {
        int canonical = -1;
        int orientation_sign = 0;
        if (!DecodeSignedDof(permutation[oriented], canonical,
                             orientation_sign) ||
            canonical >= output.scalar_face_dofs)
        {
          Require(false, "independent orientation entry", evidence);
          return false;
        }
        const int face_entry =
            component * output.scalar_face_dofs + canonical;
        if (face_entry < 0 || face_entry >= face_vdofs.Size())
        {
          Require(false, "independent face entry index", evidence);
          return false;
        }
        int carrier_vdof = -1;
        int carrier_sign = 0;
        if (!DecodeSignedDof(face_vdofs[face_entry], carrier_vdof,
                             carrier_sign) ||
            carrier_vdof >= carrier_space.GetVSize())
        {
          Require(false, "independent carrier VDof", evidence);
          return false;
        }
        const int sign = orientation_sign * carrier_sign;
        const int row = component * sc25::faces_per_quadrilateral *
                output.scalar_face_dofs +
            local_face * output.scalar_face_dofs + oriented;
        const int local_true = carrier_space.GetLocalTDofNumber(carrier_vdof);
        const auto global_true =
            carrier_space.GetGlobalTDofNumber(carrier_vdof);
        if (sign == 0 || row < 0 || row >= output.local_trace_unknowns ||
            local_true < 0 || local_true >= carrier_space.GetTrueVSize() ||
            global_true < 0 ||
            global_true >= carrier_space.GlobalTrueVSize())
        {
          Require(false, "independent true restriction index", evidence);
          return false;
        }
        const int signed_carrier_vdof =
            sign < 0 ? -1 - carrier_vdof : carrier_vdof;
        output.entries[static_cast<std::size_t>(row)] = {
            element, row, local_face, component, oriented, canonical,
            signed_carrier_vdof, carrier_vdof, sign,
            static_cast<dg::hdg::GlobalIndex>(global_true),
            static_cast<dg::hdg::LocalIndex>(local_true), 0};
      }
  }
  return true;
}

bool CompareProductionRestriction(
    const IndependentElementRestriction& expected,
    const sc25::ElementTopology& actual,
    IndependentC0Evidence& evidence)
{
  const std::size_t expected_trace_size =
      static_cast<std::size_t>(expected.local_trace_unknowns);
  if (actual.element != expected.element ||
      actual.scalar_volume_dofs != expected.scalar_volume_dofs ||
      actual.scalar_face_dofs != expected.scalar_face_dofs ||
      actual.local_interior_unknowns != expected.local_interior_unknowns ||
      actual.local_trace_unknowns != expected.local_trace_unknowns ||
      actual.faces.size() != expected.faces.size() ||
      actual.face_orientations.size() != expected.orientations.size() ||
      actual.signed_trace_vdofs.size() != expected_trace_size ||
      actual.restriction.size() != expected_trace_size ||
      actual.volume.u.size() !=
          static_cast<std::size_t>(sc25::state_components *
                                   expected.scalar_volume_dofs) ||
      actual.volume.sigma.size() !=
          static_cast<std::size_t>(expected.scalar_volume_dofs) ||
      actual.volume.z.size() !=
          static_cast<std::size_t>(sc25::flux_components *
                                   expected.scalar_volume_dofs))
  {
    Require(false, "production topology extent", evidence);
    return false;
  }
  for (std::size_t face = 0; face < expected.faces.size(); ++face)
  {
    Require(actual.faces[face] == expected.faces[face] &&
                actual.face_orientations[face] == expected.orientations[face],
            "production element edge metadata", evidence);
  }
  for (std::size_t row = 0; row < expected_trace_size; ++row)
  {
    const IndependentRestrictionEntry& left = expected.entries[row];
    const sc25::TraceRestrictionEntry& right = actual.restriction[row];
    Require(actual.signed_trace_vdofs[row] == left.signed_carrier_vdof &&
                right.element == left.element &&
                right.local_row == left.local_row &&
                right.local_face == left.local_face &&
                right.component == left.component &&
                right.oriented_face_dof == left.oriented_face_dof &&
                right.canonical_face_dof == left.canonical_face_dof &&
                right.signed_carrier_vdof == left.signed_carrier_vdof &&
                right.decoded_carrier_vdof == left.decoded_carrier_vdof &&
                right.sign == left.sign &&
                right.global_true_dof == left.global_true_dof &&
                right.local_true_dof == left.local_true_dof &&
                right.owner_rank == left.owner_rank,
            "independently reconstructed restriction row " +
                std::to_string(row),
            evidence);
    evidence.restriction.push_back(left);
    ++evidence.checked_restriction_entries;
    if (left.sign < 0) { ++evidence.negative_signed_restriction_entries; }
  }
  return true;
}

bool CompareVolumeMetadata(
    const int element, const sc25::ElementTopology& actual,
    const sc25::SpaceOwnershipSnapshot& spaces,
    IndependentC0Evidence& evidence)
{
  const auto compare = [element, &evidence](
                           const mfem::ParFiniteElementSpace* space,
                           const std::vector<int>& actual_dofs,
                           const bool actual_identity,
                           const std::string& name)
  {
    if (space == nullptr)
    {
      Require(false, name + " volume space", evidence);
      return false;
    }
    mfem::Array<int> expected_dofs;
    mfem::DofTransformation transformation(space->GetVDim(),
                                            space->GetOrdering());
    space->GetElementVDofs(element, expected_dofs, transformation);
    if (expected_dofs.Size() <= 0 ||
        actual_dofs.size() !=
            static_cast<std::size_t>(expected_dofs.Size()))
    {
      Require(false, name + " volume DOF extent", evidence);
      return false;
    }
    Require(actual_identity == transformation.IsIdentity(),
            name + " transformation identity", evidence);
    for (int entry = 0; entry < expected_dofs.Size(); ++entry)
    {
      int decoded = -1;
      int sign = 0;
      if (!DecodeSignedDof(expected_dofs[entry], decoded, sign) ||
          decoded >= space->GetVSize() || sign == 0)
      {
        Require(false, name + " volume DOF index", evidence);
        return false;
      }
      Require(actual_dofs[static_cast<std::size_t>(entry)] ==
                  expected_dofs[entry],
              name + " volume DOF entry " + std::to_string(entry),
              evidence);
    }
    return true;
  };
  return compare(spaces.u.address, actual.volume.u,
                 actual.volume.u_transformation_identity, "U") &&
      compare(spaces.sigma.address, actual.volume.sigma,
              actual.volume.sigma_transformation_identity, "Sigma") &&
      compare(spaces.z.address, actual.volume.z,
              actual.volume.z_transformation_identity, "z");
}

void IndependentEulerSigma(
    const std::array<double, 4>& conservative, const double sigma,
    const std::array<double, 2>& normal, std::array<double, 4>& flux,
    std::array<std::array<double, 5>, 4>& jacobian)
{
  const double rho = conservative[0];
  const double mx = conservative[1];
  const double my = conservative[2];
  const double energy = conservative[3];
  const double ux = mx / rho;
  const double uy = my / rho;
  const double un = ux * normal[0] + uy * normal[1];
  const double gm1 = frozen_gamma - 1.0;
  const double pressure = gm1 *
      (energy - 0.5 * (mx * mx + my * my) / rho);
  flux = {{mx * normal[0] + my * normal[1],
           mx * un + (pressure + sigma) * normal[0],
           my * un + (pressure + sigma) * normal[1],
           (energy + pressure + sigma) * un}};
  for (auto& row : jacobian) { row.fill(0.0); }
  const std::array<double, 4> pressure_derivative{{
      0.5 * gm1 * (mx * mx + my * my) / (rho * rho),
      -gm1 * mx / rho, -gm1 * my / rho, gm1}};
  const std::array<double, 4> un_derivative{{
      -un / rho, normal[0] / rho, normal[1] / rho, 0.0}};
  jacobian[0][1] = normal[0];
  jacobian[0][2] = normal[1];
  for (int column = 0; column < 4; ++column)
  {
    jacobian[1][column] = (column == 1 ? un : 0.0) +
        mx * un_derivative[column] +
        pressure_derivative[column] * normal[0];
    jacobian[2][column] = (column == 2 ? un : 0.0) +
        my * un_derivative[column] +
        pressure_derivative[column] * normal[1];
    jacobian[3][column] =
        ((column == 3 ? 1.0 : 0.0) + pressure_derivative[column]) * un +
        (energy + pressure + sigma) * un_derivative[column];
  }
  jacobian[1][4] = normal[0];
  jacobian[2][4] = normal[1];
  jacobian[3][4] = un;
}

void IndependentSourceCorrection(
    const std::array<double, 4>& conservative, const double sigma,
    const std::array<double, 2>& normal, std::array<double, 4>& correction,
    std::array<std::array<double, 5>, 4>& jacobian)
{
  const double rho = conservative[0];
  const double un = (conservative[1] * normal[0] +
                     conservative[2] * normal[1]) / rho;
  correction = {{0.0, sigma * normal[0], sigma * normal[1], sigma * un}};
  for (auto& row : jacobian) { row.fill(0.0); }
  jacobian[1][4] = normal[0];
  jacobian[2][4] = normal[1];
  jacobian[3][0] = -sigma * un / rho;
  jacobian[3][1] = sigma * normal[0] / rho;
  jacobian[3][2] = sigma * normal[1] / rho;
  jacobian[3][4] = un;
}

void CheckPointwiseAlgebra(IndependentC0Evidence& evidence)
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  Require(std::abs(sc25::gamma - frozen_gamma) <=
              ExactTolerance(frozen_gamma),
          "production frozen gamma", evidence);
  const auto initial = sc25::SmoothInitialState(0.25, 0.125);
  const double rho = 1.0 + 0.05 * std::sin(0.5 * pi) *
      std::sin(0.25 * pi);
  const double u = 0.1 * std::sin(0.5 * pi);
  const double v = 0.1 * std::sin(0.25 * pi);
  const std::array<double, 4> expected_initial{{
      rho, rho * u, rho * v,
      1.0 / (frozen_gamma - 1.0) + 0.5 * rho * (u * u + v * v)}};
  RequireExactArray(initial.conservative, expected_initial,
                    "smooth initial state", evidence);

  const std::array<std::array<double, 2>, 2> gradient{{
      {{1.0, 2.0}}, {{-3.0, 4.0}}}};
  Require(std::abs(sc25::VelocityGradientInvariant(gradient) - 30.0) <=
              ExactTolerance(30.0),
          "velocity-gradient invariant", evidence);

  const std::array<double, 4> state{{1.2, 0.3, -0.2, 3.0}};
  const std::array<double, 2> normal{{0.6, -0.8}};
  std::array<double, 4> flux{}, expected_flux{};
  std::array<std::array<double, 5>, 4> jacobian{}, expected_jacobian{};
  Require(sc25::EulerSigmaNormalFlux(state, 0.4, normal, flux, jacobian),
          "Euler-Sigma evaluation", evidence);
  IndependentEulerSigma(state, 0.4, normal, expected_flux,
                        expected_jacobian);
  RequireExactArray(flux, expected_flux, "Euler-Sigma flux", evidence);
  double expected_jacobian_norm = 0.0;
  for (const auto& row : expected_jacobian)
  {
    expected_jacobian_norm = std::max(expected_jacobian_norm, NormInf(row));
  }
  const double jacobian_tolerance = ExactTolerance(expected_jacobian_norm);
  for (int row = 0; row < sc25::state_components; ++row)
    for (int column = 0; column < sc25::trace_components; ++column)
    {
      Require(std::abs(jacobian[row][column] -
                       expected_jacobian[row][column]) <= jacobian_tolerance,
              "Euler-Sigma Jacobian (" + std::to_string(row) + "," +
                  std::to_string(column) + ")",
              evidence);
    }

  std::array<double, 4> correction{}, expected_correction{};
  std::array<std::array<double, 5>, 4> correction_jacobian{};
  std::array<std::array<double, 5>, 4> expected_correction_jacobian{};
  Require(sc25::SourceCorrectionNormalFlux(
              state, 0.4, normal, correction, correction_jacobian),
          "source correction evaluation", evidence);
  IndependentSourceCorrection(state, 0.4, normal, expected_correction,
                              expected_correction_jacobian);
  RequireExactArray(correction, expected_correction,
                    "source correction flux", evidence);
  double expected_correction_norm = 0.0;
  for (const auto& row : expected_correction_jacobian)
  {
    expected_correction_norm =
        std::max(expected_correction_norm, NormInf(row));
  }
  const double correction_tolerance = ExactTolerance(expected_correction_norm);
  for (int row = 0; row < sc25::state_components; ++row)
    for (int column = 0; column < sc25::trace_components; ++column)
    {
      Require(std::abs(correction_jacobian[row][column] -
                       expected_correction_jacobian[row][column]) <=
                  correction_tolerance,
              "source correction Jacobian (" + std::to_string(row) + "," +
                  std::to_string(column) + ")",
              evidence);
    }

  constexpr int degree = 2;
  constexpr double rhohat = 1.25;
  constexpr double height = 0.2;
  constexpr double sigma = 0.7;
  constexpr double sigmahat = 0.4;
  constexpr double z_dot_normal = -0.1;
  constexpr double alpha = 0.1;
  constexpr double test_value = 0.75;
  sc25::AuxiliaryStabilizationPoint auxiliary;
  Require(sc25::EvaluateAuxiliaryStabilization(
              degree, rhohat, height, sigma, sigmahat, z_dot_normal,
              alpha, test_value, auxiliary),
          "auxiliary stabilization evaluation", evidence);
  const double tau = 9.0 / (rhohat * height);
  const double jump = sigma - sigmahat;
  const std::array<double, 8> actual_auxiliary{{
      auxiliary.tau, auxiliary.numerical_flux,
      auxiliary.r_sigma_d_sigma, auxiliary.r_sigma_d_sigmahat,
      auxiliary.r_sigma_d_rhohat, auxiliary.g_sigma_d_sigma,
      auxiliary.g_sigma_d_sigmahat, auxiliary.g_sigma_d_rhohat}};
  const std::array<double, 8> expected_auxiliary{{
      tau, z_dot_normal - tau * jump, alpha * test_value * tau,
      -alpha * test_value * tau,
      -alpha * test_value * tau * jump / rhohat,
      -alpha * test_value * tau, alpha * test_value * tau,
      alpha * test_value * tau * jump / rhohat}};
  RequireExactArray(actual_auxiliary, expected_auxiliary,
                    "all auxiliary stabilization derivatives", evidence);

  // Isolate the v3 auxiliary face sign from every EOS, state-flux, mesh, and
  // condensation contribution.  Combining R_z(alpha*z), R_Sigma(Sigma),
  // and G_Sigma(Sigmahat) leaves alpha*(Sigma-Sigmahat)*(z.n-zhat.n).
  // The rejected v2-plus mutation changes only zhat.n's stabilization sign.
  constexpr double energy_rho = 1.0;
  constexpr double energy_sigma = 1.0;
  constexpr double energy_sigmahat = 0.0;
  constexpr double energy_z_square = 0.3125;
  constexpr double energy_z_dot_normal = 0.375;
  constexpr double energy_alpha = 0.25;
  constexpr double energy_tau = 8.0;
  constexpr double energy_jump = energy_sigma - energy_sigmahat;
  constexpr double bulk_energy =
      energy_sigma * energy_sigma / energy_rho +
      energy_alpha * energy_rho * energy_z_square;
  constexpr double v3_flux =
      energy_z_dot_normal - energy_tau * energy_jump;
  constexpr double rejected_v2_plus_flux =
      energy_z_dot_normal + energy_tau * energy_jump;
  constexpr double v3_energy = bulk_energy + energy_alpha * energy_jump *
      (energy_z_dot_normal - v3_flux);
  constexpr double rejected_v2_plus_energy = bulk_energy +
      energy_alpha * energy_jump *
          (energy_z_dot_normal - rejected_v2_plus_flux);
  Require(v3_energy == bulk_energy +
              energy_alpha * energy_tau * energy_jump * energy_jump &&
              v3_energy > 0.0,
          "positive v3 auxiliary energy identity", evidence);
  Require(rejected_v2_plus_energy < 0.0,
          "rejected v2-plus auxiliary energy mutation", evidence);

  // In scalar A/B/C/D order the v3 stabilization is
  // alpha*tau*[+1,-1;-1,+1].  Flipping only the flux sign produces the
  // rejected [-1,+1;+1,-1] pattern and its negative jump energy.
  constexpr double block_scale = energy_alpha * energy_tau;
  const std::array<double, 4> v3_blocks{{
      block_scale, -block_scale, -block_scale, block_scale}};
  const std::array<double, 4> rejected_v2_plus_blocks{{
      -block_scale, block_scale, block_scale, -block_scale}};
  const auto passes_v3_pattern = [&v3_blocks](
                                     const std::array<double, 4>& blocks)
  {
    return blocks == v3_blocks && blocks[0] > 0.0 && blocks[1] < 0.0 &&
        blocks[2] < 0.0 && blocks[3] > 0.0;
  };
  Require(passes_v3_pattern(v3_blocks),
          "v3 auxiliary A/B/C/D sign pattern", evidence);
  Require(!passes_v3_pattern(rejected_v2_plus_blocks) &&
              rejected_v2_plus_blocks[0] < 0.0 &&
              rejected_v2_plus_blocks[1] > 0.0 &&
              rejected_v2_plus_blocks[2] > 0.0 &&
              rejected_v2_plus_blocks[3] < 0.0,
          "isolated v2-plus A/B/C/D mutation must fail", evidence);
  const std::array<double, 2> trace_pair{{energy_sigma,
                                         energy_sigmahat}};
  const auto quadratic_form = [&trace_pair](
                                  const std::array<double, 4>& blocks)
  {
    return trace_pair[0] *
            (blocks[0] * trace_pair[0] + blocks[1] * trace_pair[1]) +
        trace_pair[1] *
            (blocks[2] * trace_pair[0] + blocks[3] * trace_pair[1]);
  };
  Require(quadratic_form(v3_blocks) ==
              block_scale * energy_jump * energy_jump &&
              quadratic_form(rejected_v2_plus_blocks) ==
                  -block_scale * energy_jump * energy_jump,
          "isolated auxiliary block energy sign", evidence);
}

void CheckCondensation(IndependentC0Evidence& evidence)
{
  constexpr std::array<double, 9> pivot_matrix{{
      -1.0, -2.0, -2.0,
      -2.0, -2.0, -2.0,
      -2.0, 0.0, -2.0}};
  constexpr std::array<double, 3> pivot_rhs{{1.0, 2.0, 3.0}};
  constexpr std::array<double, 3> pivot_solution{{-1.0, 0.5, -0.5}};
  constexpr std::array<int, 3> expected_pivots{{1, 2, 2}};

  const std::vector<double> pivot_matrix_vector(pivot_matrix.begin(),
                                                 pivot_matrix.end());
  const std::vector<double> pivot_rhs_vector(pivot_rhs.begin(),
                                              pivot_rhs.end());
  std::vector<double> independent_solution;
  const bool independent_solved = DirectAugmentedPartialPivotSolve(
      pivot_matrix_vector, 3, pivot_rhs_vector, independent_solution);
  Require(independent_solved, "independent multi-pivot solve", evidence);
  if (independent_solved)
  {
    RequireExactVector(independent_solution,
                       std::vector<double>(pivot_solution.begin(),
                                           pivot_solution.end()),
                       "independent multi-pivot solution", evidence);
    std::array<double, 3> application{};
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column)
      {
        application[row] += pivot_matrix[3 * row + column] *
            independent_solution[column];
      }
    RequireExactArray(application, pivot_rhs,
                      "independent multi-pivot original-matrix residual",
                      evidence);
  }

  sc25::ElementLinearization pivot_input;
  pivot_input.element = 3;
  pivot_input.local_interior_unknowns = 3;
  pivot_input.local_trace_unknowns = 3;
  pivot_input.a.assign(pivot_matrix.begin(), pivot_matrix.end());
  pivot_input.b.assign(9, 0.0);
  pivot_input.c = {1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0};
  pivot_input.d.assign(9, 0.0);
  pivot_input.local_residual.assign(pivot_rhs.begin(), pivot_rhs.end());
  pivot_input.trace_residual.assign(3, 0.0);
  sc25::ElementCondensedBlock pivot_output;
  const bool production_solved =
      sc25::CondenseElementLinearization(pivot_input, pivot_output);
  Require(production_solved, "production multi-pivot solve", evidence);
  if (production_solved)
  {
    RequireExactVector(pivot_output.right_hand_side,
                       std::vector<double>(pivot_solution.begin(),
                                           pivot_solution.end()),
                       "production multi-pivot solution", evidence);
    RequireExactVector(pivot_output.schur, std::vector<double>(9, 0.0),
                       "production multi-pivot zero Schur", evidence);
  }

  std::array<int, 3> rejected_pivots{};
  std::array<double, 3> rejected_solution{};
  const bool rejected_solved = RejectedInterleavedPartialPivotSolve3(
      pivot_matrix, pivot_rhs, rejected_pivots, rejected_solution);
  Require(rejected_solved, "rejected interleaved mutation execution",
          evidence);
  if (rejected_solved)
  {
    constexpr std::array<double, 3> expected_rejected_solution{{
        -2.0, 1.0, 0.0}};
    constexpr std::array<double, 3> expected_rejected_residual{{
        -1.0, 0.0, 1.0}};
    Require(rejected_pivots == expected_pivots,
            "multi-pivot fixture sequence", evidence);
    RequireExactArray(rejected_solution, expected_rejected_solution,
                      "rejected interleaved mutation solution", evidence);
    std::array<double, 3> rejected_residual{};
    for (int row = 0; row < 3; ++row)
    {
      rejected_residual[row] = -pivot_rhs[row];
      for (int column = 0; column < 3; ++column)
      {
        rejected_residual[row] += pivot_matrix[3 * row + column] *
            rejected_solution[column];
      }
    }
    RequireExactArray(rejected_residual, expected_rejected_residual,
                      "rejected interleaved original-matrix residual",
                      evidence);
    const double rejected_residual_norm = std::sqrt(
        rejected_residual[0] * rejected_residual[0] +
        rejected_residual[1] * rejected_residual[1] +
        rejected_residual[2] * rejected_residual[2]);
    Require(std::abs(rejected_residual_norm - std::sqrt(2.0)) <=
                ExactTolerance(std::sqrt(2.0)) &&
                rejected_residual_norm >
                    ExactTolerance(NormInf(pivot_rhs)),
            "old interleaved pivot ordering must fail", evidence);
  }

  sc25::ElementLinearization input;
  input.element = 7;
  input.local_interior_unknowns = 2;
  input.local_trace_unknowns = 2;
  input.a = {4.0, 1.0, 2.0, 3.0};
  input.b = {1.0, 2.0, 3.0, 4.0};
  input.c = {2.0, -1.0, 1.0, 1.0};
  input.d = {6.0, 0.5, -0.25, 5.0};
  input.local_residual = {2.0, -1.0};
  input.trace_residual = {0.4, -0.2};
  sc25::ElementCondensedBlock output;
  Require(sc25::CondenseElementLinearization(input, output),
          "local condensation", evidence);
  if (output.schur.size() != 4 || output.right_hand_side.size() != 2)
  {
    Require(false, "local condensation extent", evidence);
    return;
  }
  const std::array<double, 4> inverse{{0.3, -0.1, -0.2, 0.4}};
  std::array<double, 4> inverse_b{};
  std::array<double, 2> inverse_r{};
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 2; ++column)
    {
      inverse_r[row] += inverse[2 * row + column] *
          input.local_residual[column];
      for (int trace_column = 0; trace_column < 2; ++trace_column)
      {
        inverse_b[2 * row + trace_column] +=
            inverse[2 * row + column] * input.b[2 * column + trace_column];
      }
    }
  std::vector<double> expected_schur(4, 0.0);
  std::vector<double> expected_rhs(2, 0.0);
  for (int row = 0; row < 2; ++row)
  {
    expected_rhs[row] = -input.trace_residual[row];
    for (int inner = 0; inner < 2; ++inner)
    {
      expected_rhs[row] += input.c[2 * row + inner] * inverse_r[inner];
    }
    for (int column = 0; column < 2; ++column)
    {
      expected_schur[2 * row + column] = input.d[2 * row + column];
      for (int inner = 0; inner < 2; ++inner)
      {
        expected_schur[2 * row + column] -=
            input.c[2 * row + inner] * inverse_b[2 * inner + column];
      }
    }
  }
  RequireExactVector(output.schur, expected_schur,
                     "Schur complement", evidence);
  RequireExactVector(output.right_hand_side, expected_rhs,
                     "condensed residual", evidence);
}

void FillField(mfem::ParGridFunction& field, const double base,
               const double stride)
{
  double* values = field.HostWrite();
  for (int entry = 0; entry < field.Size(); ++entry)
  {
    values[entry] = base + stride * entry;
  }
}

bool IndependentGather(const int element,
                       const mfem::ParFiniteElementSpace& space,
                       const mfem::ParGridFunction& field,
                       std::vector<double>& output)
{
  mfem::Array<int> vdofs;
  mfem::DofTransformation transformation(space.GetVDim(),
                                          space.GetOrdering());
  space.GetElementVDofs(element, vdofs, transformation);
  if (vdofs.Size() <= 0) { return false; }
  mfem::Vector local(vdofs.Size());
  field.GetSubVector(vdofs, local);
  transformation.InvTransformPrimal(local);
  output.assign(local.HostRead(), local.HostRead() + local.Size());
  return true;
}

bool IndependentAddDual(const int element,
                        const mfem::ParFiniteElementSpace& space,
                        const std::vector<double>& local_values,
                        mfem::ParGridFunction& destination)
{
  mfem::Array<int> vdofs;
  mfem::DofTransformation transformation(space.GetVDim(),
                                          space.GetOrdering());
  space.GetElementVDofs(element, vdofs, transformation);
  if (vdofs.Size() <= 0 ||
      local_values.size() != static_cast<std::size_t>(vdofs.Size()))
  {
    return false;
  }
  mfem::Vector local(vdofs.Size());
  std::copy(local_values.begin(), local_values.end(), local.HostWrite());
  transformation.TransformDual(local);
  destination.AddElementVector(vdofs, local);
  return true;
}

void CompareGridFunctions(const mfem::ParGridFunction& actual,
                          const mfem::ParGridFunction& expected,
                          const std::string& message,
                          IndependentC0Evidence& evidence)
{
  if (actual.Size() != expected.Size())
  {
    Require(false, message + " extent", evidence);
    return;
  }
  const double* actual_values = actual.HostRead();
  const double* expected_values = expected.HostRead();
  double norm = 0.0;
  for (int entry = 0; entry < expected.Size(); ++entry)
  {
    norm = std::max(norm, std::abs(expected_values[entry]));
  }
  const double tolerance = ExactTolerance(norm);
  for (int entry = 0; entry < expected.Size(); ++entry)
  {
    Require(std::abs(actual_values[entry] - expected_values[entry]) <=
                tolerance,
            message + " entry " + std::to_string(entry), evidence);
  }
}

} // namespace

bool RunIndependentC0Oracle(
    sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    IndependentC0Evidence& evidence)
{
  evidence = {};
  const mfem::ParMesh* mesh = foundation.mesh();
  const auto* volume_collection = foundation.volume_collection();
  const auto* trace_collection = foundation.trace_collection();
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  const sc25::FieldOwnershipView fields = foundation.fields();
  Require(foundation.status() == sc25::FoundationStatus::ready,
          "foundation ready", evidence);
  if (mesh == nullptr || volume_collection == nullptr ||
      trace_collection == nullptr || spaces.u.address == nullptr ||
      spaces.sigma.address == nullptr || spaces.z.address == nullptr ||
      spaces.uhat.address == nullptr || spaces.sigmahat.address == nullptr ||
      spaces.trace_carrier.address == nullptr || fields.u == nullptr ||
      fields.sigma == nullptr || fields.z == nullptr || fields.uhat == nullptr ||
      fields.sigmahat == nullptr || fields.trace_carrier == nullptr)
  {
    Require(false, "complete MFEM ownership graph", evidence);
    return false;
  }

  CheckSpace(spaces.u, mesh, volume_collection, 4, "U", evidence);
  CheckSpace(spaces.sigma, mesh, volume_collection, 1, "Sigma", evidence);
  CheckSpace(spaces.z, mesh, volume_collection, 2, "z", evidence);
  CheckSpace(spaces.uhat, mesh, trace_collection, 4, "Uhat", evidence);
  CheckSpace(spaces.sigmahat, mesh, trace_collection, 1, "Sigmahat",
             evidence);
  CheckSpace(spaces.trace_carrier, mesh, trace_collection, 5,
             "trace carrier", evidence);
  const std::array<const mfem::ParFiniteElementSpace*, 6> addresses{{
      spaces.u.address, spaces.sigma.address, spaces.z.address,
      spaces.uhat.address, spaces.sigmahat.address,
      spaces.trace_carrier.address}};
  for (std::size_t left = 0; left < addresses.size(); ++left)
    for (std::size_t right = left + 1; right < addresses.size(); ++right)
    {
      Require(addresses[left] != addresses[right], "distinct MFEM spaces",
              evidence);
    }
  Require(spaces.u.local_vector_size == 4 * spaces.sigma.local_vector_size &&
              spaces.z.local_vector_size ==
                  2 * spaces.sigma.local_vector_size &&
              spaces.uhat.local_vector_size ==
                  4 * spaces.sigmahat.local_vector_size &&
              spaces.trace_carrier.local_vector_size ==
                  5 * spaces.sigmahat.local_vector_size,
          "scalar local ownership ratios", evidence);
  Require(spaces.u.local_true_size == 4 * spaces.sigma.local_true_size &&
              spaces.z.local_true_size == 2 * spaces.sigma.local_true_size &&
              spaces.uhat.local_true_size ==
                  4 * spaces.sigmahat.local_true_size &&
              spaces.trace_carrier.local_true_size ==
                  5 * spaces.sigmahat.local_true_size &&
              spaces.u.global_true_size ==
                  4 * spaces.sigma.global_true_size &&
              spaces.z.global_true_size ==
                  2 * spaces.sigma.global_true_size &&
              spaces.uhat.global_true_size ==
                  4 * spaces.sigmahat.global_true_size &&
              spaces.trace_carrier.global_true_size ==
                  5 * spaces.sigmahat.global_true_size,
          "scalar true ownership ratios", evidence);
  Require(fields.u->ParFESpace() == spaces.u.address &&
              fields.sigma->ParFESpace() == spaces.sigma.address &&
              fields.z->ParFESpace() == spaces.z.address &&
              fields.uhat->ParFESpace() == spaces.uhat.address &&
              fields.sigmahat->ParFESpace() == spaces.sigmahat.address &&
              fields.trace_carrier->ParFESpace() ==
                  spaces.trace_carrier.address,
          "separate field ownership", evidence);

  const auto& topology = foundation.element_topology();
  if (topology.size() != static_cast<std::size_t>(mesh->GetNE()))
  {
    Require(false, "production topology extent", evidence);
    return false;
  }
  std::vector<IndependentElementRestriction> independent_topology(
      static_cast<std::size_t>(mesh->GetNE()));
  for (int element = 0; element < mesh->GetNE(); ++element)
  {
    if (!BuildIndependentRestriction(
            element, foundation.config().polynomial_degree, *mesh,
            *trace_collection, *spaces.trace_carrier.address,
            independent_topology[static_cast<std::size_t>(element)],
            evidence) ||
        !CompareProductionRestriction(
            independent_topology[static_cast<std::size_t>(element)],
            topology[static_cast<std::size_t>(element)], evidence) ||
        !CompareVolumeMetadata(
            element, topology[static_cast<std::size_t>(element)], spaces,
            evidence))
    {
      return false;
    }
    ++evidence.checked_elements;
  }

  const int carrier_true_size = spaces.trace_carrier.local_true_size;
  const int carrier_local_size = spaces.trace_carrier.local_vector_size;
  const int scalar_trace_size = spaces.sigmahat.local_vector_size;
  if (carrier_true_size <= 0 || carrier_local_size != 5 * scalar_trace_size ||
      fields.trace_carrier->Size() != carrier_local_size ||
      fields.uhat->Size() != 4 * scalar_trace_size ||
      fields.sigmahat->Size() != scalar_trace_size)
  {
    Require(false, "trace field extents", evidence);
    return false;
  }
  mfem::Vector true_values(carrier_true_size);
  for (int entry = 0; entry < true_values.Size(); ++entry)
  {
    true_values[entry] = 0.125 + 0.001 * entry;
  }
  if (!foundation.SetTraceCarrierFromTrueDofs(true_values))
  {
    Require(false, "SetFromTrueDofs recovery", evidence);
    return false;
  }
  const double* carrier_values = fields.trace_carrier->HostRead();
  const double* uhat_values = fields.uhat->HostRead();
  const double* sigmahat_values = fields.sigmahat->HostRead();
  for (int local = 0; local < carrier_local_size; ++local)
  {
    const int local_true =
        spaces.trace_carrier.address->GetLocalTDofNumber(local);
    if (local_true < 0 || local_true >= true_values.Size())
    {
      Require(false, "carrier local-to-true recovery index", evidence);
      return false;
    }
    Require(std::abs(carrier_values[local] - true_values[local_true]) <=
                ExactTolerance(std::abs(true_values[local_true])),
            "carrier SetFromTrueDofs value", evidence);
  }
  for (int scalar = 0; scalar < scalar_trace_size; ++scalar)
  {
    for (int component = 0; component < 4; ++component)
    {
      const int carrier_entry = component * scalar_trace_size + scalar;
      Require(std::abs(uhat_values[carrier_entry] -
                       carrier_values[carrier_entry]) <=
                  ExactTolerance(std::abs(carrier_values[carrier_entry])),
              "Uhat unpack value", evidence);
    }
    const int carrier_entry = 4 * scalar_trace_size + scalar;
    Require(std::abs(sigmahat_values[scalar] - carrier_values[carrier_entry]) <=
                ExactTolerance(std::abs(carrier_values[carrier_entry])),
            "Sigmahat unpack value", evidence);
  }

  auto* mutable_uhat = const_cast<mfem::ParGridFunction*>(fields.uhat);
  auto* mutable_sigmahat =
      const_cast<mfem::ParGridFunction*>(fields.sigmahat);
  FillField(*mutable_uhat, -0.4, 0.0003);
  FillField(*mutable_sigmahat, 0.7, -0.0002);
  std::vector<double> expected_packed(
      static_cast<std::size_t>(carrier_local_size));
  std::copy(mutable_uhat->HostRead(),
            mutable_uhat->HostRead() + mutable_uhat->Size(),
            expected_packed.begin());
  std::copy(mutable_sigmahat->HostRead(),
            mutable_sigmahat->HostRead() + mutable_sigmahat->Size(),
            expected_packed.begin() + 4 * scalar_trace_size);
  if (!foundation.PackTraceCarrierFromPhysicalFields())
  {
    Require(false, "pack physical trace fields", evidence);
    return false;
  }
  carrier_values = fields.trace_carrier->HostRead();
  std::vector<double> actual_packed(carrier_values,
                                    carrier_values + carrier_local_size);
  RequireExactVector(actual_packed, expected_packed, "packed trace carrier",
                     evidence);
  if (!foundation.SetTraceCarrierFromTrueDofs(true_values))
  {
    Require(false, "restore carrier true values", evidence);
    return false;
  }

  for (const IndependentElementRestriction& element : independent_topology)
  {
    mfem::Array<int> signed_dofs(element.local_trace_unknowns);
    for (int row = 0; row < signed_dofs.Size(); ++row)
    {
      signed_dofs[row] = element.entries[static_cast<std::size_t>(row)]
                             .signed_carrier_vdof;
    }
    mfem::Vector local(signed_dofs.Size());
    fields.trace_carrier->GetSubVector(signed_dofs, local);
    for (int row = 0; row < local.Size(); ++row)
    {
      const IndependentRestrictionEntry& entry =
          element.entries[static_cast<std::size_t>(row)];
      const double expected = entry.sign * true_values[entry.local_true_dof];
      evidence.maximum_trace_action_error = std::max(
          evidence.maximum_trace_action_error, std::abs(local[row] - expected));
    }
  }
  Require(evidence.maximum_trace_action_error <=
              ExactTolerance(NormInf(std::vector<double>(
                  true_values.HostRead(),
                  true_values.HostRead() + true_values.Size()))),
          "signed restriction action", evidence);

  auto* mutable_u = const_cast<mfem::ParGridFunction*>(fields.u);
  auto* mutable_sigma = const_cast<mfem::ParGridFunction*>(fields.sigma);
  auto* mutable_z = const_cast<mfem::ParGridFunction*>(fields.z);
  FillField(*mutable_u, 0.2, 0.0007);
  FillField(*mutable_sigma, -0.3, 0.0009);
  FillField(*mutable_z, 0.4, -0.0005);
  sc25::ElementFieldCoefficients gathered;
  std::vector<double> expected_u, expected_sigma, expected_z;
  if (!IndependentGather(0, *spaces.u.address, *fields.u, expected_u) ||
      !IndependentGather(0, *spaces.sigma.address, *fields.sigma,
                         expected_sigma) ||
      !IndependentGather(0, *spaces.z.address, *fields.z, expected_z))
  {
    Require(false, "independent primal gather", evidence);
    return false;
  }
  if (!foundation.GatherElementFields(0, gathered))
  {
    Require(false, "production primal gather", evidence);
    return false;
  }
  RequireExactVector(gathered.u, expected_u, "nonzero U primal gather",
                     evidence);
  RequireExactVector(gathered.sigma, expected_sigma,
                     "nonzero Sigma primal gather", evidence);
  RequireExactVector(gathered.z, expected_z, "nonzero z primal gather",
                     evidence);

  mfem::ParGridFunction u_dual(
      const_cast<mfem::ParFiniteElementSpace*>(spaces.u.address));
  mfem::ParGridFunction sigma_dual(
      const_cast<mfem::ParFiniteElementSpace*>(spaces.sigma.address));
  mfem::ParGridFunction z_dual(
      const_cast<mfem::ParFiniteElementSpace*>(spaces.z.address));
  mfem::ParGridFunction expected_u_dual(
      const_cast<mfem::ParFiniteElementSpace*>(spaces.u.address));
  mfem::ParGridFunction expected_sigma_dual(
      const_cast<mfem::ParFiniteElementSpace*>(spaces.sigma.address));
  mfem::ParGridFunction expected_z_dual(
      const_cast<mfem::ParFiniteElementSpace*>(spaces.z.address));
  u_dual = 0.0;
  sigma_dual = 0.0;
  z_dual = 0.0;
  expected_u_dual = 0.0;
  expected_sigma_dual = 0.0;
  expected_z_dual = 0.0;
  sc25::ElementFieldCoefficients local_dual;
  local_dual.u.resize(expected_u.size());
  local_dual.sigma.resize(expected_sigma.size());
  local_dual.z.resize(expected_z.size());
  for (std::size_t entry = 0; entry < local_dual.u.size(); ++entry)
  {
    local_dual.u[entry] = 0.25 + 0.01 * entry;
  }
  for (std::size_t entry = 0; entry < local_dual.sigma.size(); ++entry)
  {
    local_dual.sigma[entry] = -0.5 + 0.02 * entry;
  }
  for (std::size_t entry = 0; entry < local_dual.z.size(); ++entry)
  {
    local_dual.z[entry] = 0.75 - 0.015 * entry;
  }
  if (!IndependentAddDual(0, *spaces.u.address, local_dual.u,
                          expected_u_dual) ||
      !IndependentAddDual(0, *spaces.sigma.address, local_dual.sigma,
                          expected_sigma_dual) ||
      !IndependentAddDual(0, *spaces.z.address, local_dual.z,
                          expected_z_dual))
  {
    Require(false, "independent dual insertion", evidence);
    return false;
  }
  if (!foundation.AddElementDualResidual(
          0, local_dual, u_dual, sigma_dual, z_dual))
  {
    Require(false, "production dual insertion", evidence);
    return false;
  }
  CompareGridFunctions(u_dual, expected_u_dual,
                       "transformed U dual output", evidence);
  CompareGridFunctions(sigma_dual, expected_sigma_dual,
                       "transformed Sigma dual output", evidence);
  CompareGridFunctions(z_dual, expected_z_dual,
                       "transformed z dual output", evidence);
  const mfem::Vector retained_u_dual(u_dual);
  const mfem::Vector retained_sigma_dual(sigma_dual);
  const mfem::Vector retained_z_dual(z_dual);
  sc25::ElementFieldCoefficients malformed_dual = local_dual;
  malformed_dual.sigma.pop_back();
  Require(!foundation.AddElementDualResidual(
              0, malformed_dual, u_dual, sigma_dual, z_dual),
          "malformed dual insertion rejected", evidence);
  for (int entry = 0; entry < u_dual.Size(); ++entry)
  {
    Require(u_dual[entry] == retained_u_dual[entry],
            "failed dual insertion preserves U", evidence);
  }
  for (int entry = 0; entry < sigma_dual.Size(); ++entry)
  {
    Require(sigma_dual[entry] == retained_sigma_dual[entry],
            "failed dual insertion preserves Sigma", evidence);
  }
  for (int entry = 0; entry < z_dual.Size(); ++entry)
  {
    Require(z_dual[entry] == retained_z_dual[entry],
            "failed dual insertion preserves z", evidence);
  }

  std::vector<sc25::ElementCondensedBlock> blocks(
      independent_topology.size());
  std::vector<double> expected_action(
      static_cast<std::size_t>(carrier_true_size), 0.0);
  std::vector<double> expected_rhs(expected_action.size(), 0.0);
  std::set<std::pair<int, int>> expected_local_pattern;
  for (std::size_t element = 0; element < independent_topology.size();
       ++element)
  {
    const IndependentElementRestriction& restriction =
        independent_topology[element];
    const int nt = restriction.local_trace_unknowns;
    blocks[element].element = static_cast<int>(element);
    blocks[element].local_trace_unknowns = nt;
    blocks[element].schur.assign(static_cast<std::size_t>(nt) * nt, 0.0);
    blocks[element].right_hand_side.resize(static_cast<std::size_t>(nt));
    for (int row = 0; row < nt; ++row)
    {
      blocks[element].schur[static_cast<std::size_t>(row) * nt + row] =
          1.5 + 0.002 * static_cast<double>(element) + 0.0001 * row;
      blocks[element].schur[static_cast<std::size_t>(row) * nt +
                            (row + 1) % nt] =
          0.01 * (1 + row % 7);
      blocks[element].schur[static_cast<std::size_t>(row) * nt +
                            (row + 9) % nt] =
          -0.003 * (1 + row % 5);
      blocks[element].right_hand_side[row] =
          0.0001 * (1 + static_cast<int>(element)) + 0.000001 * row;
    }
    for (int row = 0; row < nt; ++row)
      for (int column = 0; column < nt; ++column)
      {
        if (blocks[element].schur[
                static_cast<std::size_t>(row) * nt + column] != 0.0)
        {
          expected_local_pattern.emplace(
              restriction.entries[static_cast<std::size_t>(row)]
                  .decoded_carrier_vdof,
              restriction.entries[static_cast<std::size_t>(column)]
                  .decoded_carrier_vdof);
        }
      }
    std::vector<double> local_x(static_cast<std::size_t>(nt), 0.0);
    for (int column = 0; column < nt; ++column)
    {
      const IndependentRestrictionEntry& entry =
          restriction.entries[static_cast<std::size_t>(column)];
      local_x[column] = entry.sign * true_values[entry.local_true_dof];
    }
    for (int row = 0; row < nt; ++row)
    {
      double local_action = 0.0;
      for (int column = 0; column < nt; ++column)
      {
        local_action += blocks[element].schur[
            static_cast<std::size_t>(row) * nt + column] * local_x[column];
      }
      const IndependentRestrictionEntry& entry =
          restriction.entries[static_cast<std::size_t>(row)];
      expected_action[entry.local_true_dof] += entry.sign * local_action;
      expected_rhs[entry.local_true_dof] +=
          entry.sign * blocks[element].right_hand_side[row];
    }
  }
  Require(foundation.assembled_trace_identity().successful_generation == 0,
          "initial assembly generation", evidence);
  if (!foundation.AssembleTrueTraceSystem(blocks))
  {
    Require(false, "MFEM true trace assembly", evidence);
    return false;
  }
  const mfem::HypreParMatrix* matrix =
      foundation.assembled_true_trace_matrix();
  const mfem::Vector* rhs =
      foundation.assembled_true_trace_right_hand_side();
  const sc25::TraceAssemblyIdentity identity =
      foundation.assembled_trace_identity();
  if (matrix == nullptr || rhs == nullptr || rhs->Size() != carrier_true_size ||
      identity.successful_generation != 1 ||
      identity.local_height != carrier_true_size ||
      identity.local_width != carrier_true_size ||
      identity.local_true_size != carrier_true_size ||
      identity.global_true_size != spaces.trace_carrier.global_true_size ||
      identity.local_nonzeros != expected_local_pattern.size())
  {
    Require(false, "retained true trace system identity", evidence);
    return false;
  }
  mfem::Vector action(carrier_true_size);
  matrix->Mult(true_values, action);
  evidence.true_action_relative_l2 = RelativeL2(
      action.HostRead(), expected_action.data(), carrier_true_size);
  evidence.true_rhs_relative_l2 = RelativeL2(
      rhs->HostRead(), expected_rhs.data(), carrier_true_size);
  Require(evidence.true_action_relative_l2 <= assembled_action_tolerance,
          "independent nonsymmetric P^T S P normalized-L2 action", evidence);
  Require(evidence.true_rhs_relative_l2 <= assembled_action_tolerance,
          "independent P^T RHS normalized-L2 assembly", evidence);

  const mfem::HypreParMatrix* retained_matrix = matrix;
  const mfem::Vector* retained_rhs = rhs;
  const mfem::Vector retained_rhs_values(*rhs);
  const mfem::Vector retained_matrix_action(action);
  mfem::Vector retention_probe(carrier_true_size);
  for (int entry = 0; entry < retention_probe.Size(); ++entry)
  {
    retention_probe[entry] =
        (entry % 2 == 0 ? 1.0 : -1.0) * (0.2 + 0.0007 * entry);
  }
  mfem::Vector retained_probe_action(carrier_true_size);
  retained_matrix->Mult(retention_probe, retained_probe_action);
  std::vector<sc25::ElementCondensedBlock> malformed_blocks = blocks;
  malformed_blocks.front().right_hand_side.pop_back();
  if (foundation.AssembleTrueTraceSystem(malformed_blocks))
  {
    Require(false, "malformed true assembly rejected", evidence);
    return false;
  }
  if (foundation.assembled_true_trace_matrix() != retained_matrix ||
      foundation.assembled_true_trace_right_hand_side() != retained_rhs ||
      foundation.assembled_trace_identity().successful_generation !=
          identity.successful_generation)
  {
    Require(false, "failed assembly preserves retained generation", evidence);
    return false;
  }
  const mfem::Vector* preserved_rhs =
      foundation.assembled_true_trace_right_hand_side();
  if (preserved_rhs == nullptr ||
      preserved_rhs->Size() != retained_rhs_values.Size())
  {
    Require(false, "preserved RHS extent", evidence);
    return false;
  }
  for (int entry = 0; entry < retained_rhs_values.Size(); ++entry)
  {
    Require((*preserved_rhs)[entry] == retained_rhs_values[entry],
            "failed assembly preserves RHS bytes", evidence);
  }
  Require(RelativeL2(preserved_rhs->HostRead(),
                     retained_rhs_values.HostRead(),
                     retained_rhs_values.Size()) <=
              assembled_action_tolerance,
          "failed assembly preserves normalized-L2 RHS", evidence);
  mfem::Vector preserved_matrix_action(carrier_true_size);
  mfem::Vector preserved_probe_action(carrier_true_size);
  foundation.assembled_true_trace_matrix()->Mult(
      true_values, preserved_matrix_action);
  foundation.assembled_true_trace_matrix()->Mult(
      retention_probe, preserved_probe_action);
  const double retained_primary_relative_l2 = RelativeL2(
      preserved_matrix_action.HostRead(), retained_matrix_action.HostRead(),
      carrier_true_size);
  const double retained_probe_relative_l2 = RelativeL2(
      preserved_probe_action.HostRead(), retained_probe_action.HostRead(),
      carrier_true_size);
  evidence.retained_matrix_relative_l2 =
      std::max(retained_primary_relative_l2, retained_probe_relative_l2);
  Require(retained_primary_relative_l2 <= assembled_action_tolerance,
          "failed assembly preserves primary matrix action", evidence);
  Require(retained_probe_relative_l2 <= assembled_action_tolerance,
          "failed assembly preserves independent probe matrix action",
          evidence);
  evidence.retained_assembly_generation = identity.successful_generation;

  CheckPointwiseAlgebra(evidence);
  CheckCondensation(evidence);
  return evidence.failures.empty();
}

namespace
{

constexpr int c1_dimension = 2;
constexpr int c1_elements_per_direction = 3;
constexpr int c1_polynomial_degree = 1;
constexpr int c1_state_components = 4;
constexpr int c1_local_components = 7;
constexpr int c1_trace_components = 5;
constexpr int c1_faces_per_element = 4;
constexpr int c1_elements =
    c1_elements_per_direction * c1_elements_per_direction;
constexpr int c1_faces = c1_dimension * c1_elements;
constexpr int c1_ns = (c1_polynomial_degree + 1) *
    (c1_polynomial_degree + 1);
constexpr int c1_face_nodes = c1_polynomial_degree + 1;
constexpr int c1_nf = c1_faces_per_element * c1_face_nodes;
constexpr int c1_ni = c1_local_components * c1_ns;
constexpr int c1_nt = c1_trace_components * c1_nf;
constexpr int c1_global_interior = c1_elements * c1_ni;
constexpr int c1_true = c1_trace_components * c1_faces * c1_face_nodes;
constexpr int c1_restriction_rows = c1_elements * c1_nt;
constexpr int c1_full = c1_global_interior + c1_true;
constexpr int c1_element_variables = c1_ni + c1_nt;
constexpr int c1_quadrature_order = 4 * c1_polynomial_degree + 8;
constexpr int c1_repeat_quadrature_order =
    4 * c1_polynomial_degree + 10;
constexpr double c1_dt = 1.0 / 4096.0;
static_assert(c1_elements == 9 && c1_faces == 18 && c1_ns == 4 &&
                  c1_nf == 8 && c1_ni == 28 && c1_nt == 40 &&
                  c1_global_interior == 252 && c1_true == 180 &&
                  c1_restriction_rows == 360 && c1_full == 432 &&
                  c1_quadrature_order == 12 &&
                  c1_repeat_quadrature_order == 14 &&
                  c1_element_variables == 68,
              "independent C1-MFEM-PERIODIC-02 dimensions must remain exact");

constexpr int n8_elements = 64;
constexpr int n8_faces = 128;
constexpr int n8_true_trace = 1280;
constexpr int n8_steps = 20;
constexpr int n8_newton_limit = 20;
constexpr int n8_line_trials = 13;
constexpr int n8_q12 = 12;
constexpr int n8_q14 = 14;
constexpr int n8_trace_restart_dimension = 50;
constexpr int n8_trace_maximum_iterations = 200;
constexpr double n8_dt = 1.0 / 4096.0;
constexpr double n8_conservation_maximum_tolerance = 1.0e-11;
constexpr double n8_conservation_negative_offset = 4.0e-11;
static_assert(sc25::persistent_elements == n8_elements &&
                  sc25::persistent_unique_faces == n8_faces &&
                  sc25::persistent_true_trace_unknowns == n8_true_trace &&
                  sc25::persistent_required_steps == n8_steps &&
                  sc25::persistent_newton_maximum_iterations ==
                      n8_newton_limit &&
                  sc25::persistent_line_search_trials == n8_line_trials &&
                  sc25::coupled_fixture_quadrature_order == n8_q12 &&
                  sc25::coupled_fixture_repeat_quadrature_order == n8_q14 &&
                  sc25::persistent_trace_restart_dimension ==
                      n8_trace_restart_dimension &&
                  sc25::persistent_trace_maximum_iterations ==
                      n8_trace_maximum_iterations &&
                  sc25::persistent_dt == n8_dt,
              "independent N8 trajectory constants must detect drift");

bool ConservationMaximumMatches(const double independent_q14,
                                const double reported_q12) noexcept
{
  return std::isfinite(independent_q14) && std::isfinite(reported_q12) &&
      std::abs(independent_q14 - reported_q12) <=
          n8_conservation_maximum_tolerance;
}

constexpr int n8_attempt_capacity = n8_steps + 4;

struct AttemptStepMapping final
{
  int step_count = 0;
  int attempt_count = 0;
  std::array<int, n8_steps> step_numbers{};
  std::array<int, n8_steps> step_attempt_indices{};
  std::array<sc25::TrajectoryStatus, n8_attempt_capacity> attempt_status{};
  std::array<int, n8_attempt_capacity> attempt_target_steps{};
};

bool AcceptedAttemptStatus(const sc25::TrajectoryStatus status) noexcept
{
  return status == sc25::TrajectoryStatus::HdgSc25Accepted ||
      status == sc25::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry;
}

bool ValidAcceptedAttemptStepBijection(
    const AttemptStepMapping& mapping,
    const std::uint64_t checkpoint_offset) noexcept
{
  if (checkpoint_offset > static_cast<std::uint64_t>(n8_steps) ||
      mapping.step_count !=
          n8_steps - static_cast<int>(checkpoint_offset) ||
      mapping.attempt_count < mapping.step_count ||
      mapping.attempt_count > n8_attempt_capacity)
  {
    return false;
  }
  std::array<int, n8_attempt_capacity> use_count{};
  for (int local_step = 0; local_step < mapping.step_count; ++local_step)
  {
    const int expected_step =
        static_cast<int>(checkpoint_offset) + local_step + 1;
    const int attempt_index = mapping.step_attempt_indices[
        static_cast<std::size_t>(local_step)];
    if (mapping.step_numbers[static_cast<std::size_t>(local_step)] !=
            expected_step ||
        attempt_index < 0 || attempt_index >= mapping.attempt_count ||
        !AcceptedAttemptStatus(mapping.attempt_status[
            static_cast<std::size_t>(attempt_index)]) ||
        mapping.attempt_target_steps[
            static_cast<std::size_t>(attempt_index)] != expected_step)
    {
      return false;
    }
    ++use_count[static_cast<std::size_t>(attempt_index)];
  }
  for (int attempt_index = 0; attempt_index < mapping.attempt_count;
       ++attempt_index)
  {
    const int target_step = mapping.attempt_target_steps[
        static_cast<std::size_t>(attempt_index)];
    const bool relevant =
        target_step > static_cast<int>(checkpoint_offset) &&
        target_step <= n8_steps;
    const bool relevant_accepted = relevant && AcceptedAttemptStatus(
        mapping.attempt_status[static_cast<std::size_t>(attempt_index)]);
    const int uses = use_count[static_cast<std::size_t>(attempt_index)];
    if (!relevant_accepted)
    {
      if (uses != 0) { return false; }
      continue;
    }
    const int local_step =
        target_step - static_cast<int>(checkpoint_offset) - 1;
    if (uses != 1 || local_step < 0 || local_step >= mapping.step_count ||
        mapping.step_attempt_indices[static_cast<std::size_t>(local_step)] !=
            attempt_index ||
        mapping.step_numbers[static_cast<std::size_t>(local_step)] !=
            target_step)
    {
      return false;
    }
  }
  return true;
}

struct C1Dual final
{
  double value = 0.0;
  std::array<double, c1_element_variables> derivative{};

  C1Dual() = default;
  C1Dual(const double input) : value(input) {}
  static C1Dual Variable(const double input, const int index)
  {
    C1Dual result(input);
    result.derivative[static_cast<std::size_t>(index)] = 1.0;
    return result;
  }
};

C1Dual operator+(const C1Dual& left, const C1Dual& right)
{
  C1Dual result(left.value + right.value);
  for (int entry = 0; entry < c1_element_variables; ++entry)
  {
    result.derivative[entry] =
        left.derivative[entry] + right.derivative[entry];
  }
  return result;
}

C1Dual operator-(const C1Dual& left, const C1Dual& right)
{
  C1Dual result(left.value - right.value);
  for (int entry = 0; entry < c1_element_variables; ++entry)
  {
    result.derivative[entry] =
        left.derivative[entry] - right.derivative[entry];
  }
  return result;
}

C1Dual operator*(const C1Dual& left, const C1Dual& right)
{
  C1Dual result(left.value * right.value);
  for (int entry = 0; entry < c1_element_variables; ++entry)
  {
    result.derivative[entry] =
        left.derivative[entry] * right.value +
        left.value * right.derivative[entry];
  }
  return result;
}

C1Dual operator/(const C1Dual& left, const C1Dual& right)
{
  C1Dual result(left.value / right.value);
  const double denominator = right.value * right.value;
  for (int entry = 0; entry < c1_element_variables; ++entry)
  {
    result.derivative[entry] =
        (left.derivative[entry] * right.value -
         left.value * right.derivative[entry]) / denominator;
  }
  return result;
}

C1Dual& operator+=(C1Dual& left, const C1Dual& right)
{
  left = left + right;
  return left;
}

C1Dual& operator-=(C1Dual& left, const C1Dual& right)
{
  left = left - right;
  return left;
}

double C1Value(const double value) { return value; }
double C1Value(const C1Dual& value) { return value.value; }

template <typename Scalar>
Scalar C1Evaluate(const std::vector<Scalar>& coefficients,
                  const int component, const int scalar_dofs,
                  const mfem::Vector& shape)
{
  Scalar value = 0.0;
  for (int scalar = 0; scalar < scalar_dofs; ++scalar)
  {
    value += coefficients[static_cast<std::size_t>(component) * scalar_dofs +
                          scalar] * shape[scalar];
  }
  return value;
}

template <typename Scalar>
Scalar C1Derivative(const std::vector<Scalar>& coefficients,
                    const int component, const int scalar_dofs,
                    const mfem::DenseMatrix& derivative,
                    const int direction)
{
  Scalar value = 0.0;
  for (int scalar = 0; scalar < scalar_dofs; ++scalar)
  {
    value += coefficients[static_cast<std::size_t>(component) * scalar_dofs +
                          scalar] * derivative(scalar, direction);
  }
  return value;
}

template <typename Scalar>
bool C1Admissible(const std::array<Scalar, 4>& state)
{
  const double rho = C1Value(state[0]);
  const double mx = C1Value(state[1]);
  const double my = C1Value(state[2]);
  const double energy = C1Value(state[3]);
  if (!std::isfinite(rho) || !std::isfinite(mx) || !std::isfinite(my) ||
      !std::isfinite(energy) || rho < 1.0e-8)
  {
    return false;
  }
  const double pressure = (frozen_gamma - 1.0) *
      (energy - 0.5 * (mx * mx + my * my) / rho);
  const double speed = std::sqrt(mx * mx + my * my) / rho;
  const double sound = std::sqrt(frozen_gamma * pressure / rho);
  return std::isfinite(pressure) && pressure >= 1.0e-8 &&
      std::isfinite(speed) && std::isfinite(sound) &&
      speed + sound < 2.5;
}

template <typename Scalar>
std::array<Scalar, 4> C1Flux(
    const std::array<Scalar, 4>& state, const Scalar& sigma,
    const std::array<double, 2>& normal)
{
  const Scalar rho = state[0];
  const Scalar mx = state[1];
  const Scalar my = state[2];
  const Scalar energy = state[3];
  const Scalar pressure = (frozen_gamma - 1.0) *
      (energy - 0.5 * (mx * mx + my * my) / rho);
  const Scalar un = (mx * normal[0] + my * normal[1]) / rho;
  return {{mx * normal[0] + my * normal[1],
           mx * un + (pressure + sigma) * normal[0],
           my * un + (pressure + sigma) * normal[1],
           (energy + pressure + sigma) * un}};
}

double C1ElementMeasure(mfem::ParMesh& mesh, const int element,
                        const int order)
{
  mfem::ElementTransformation* transformation =
      mesh.GetElementTransformation(element);
  if (transformation == nullptr) { return 0.0; }
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(
      mesh.GetElementGeometry(element), order);
  double result = 0.0;
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = rule.IntPoint(point);
    transformation->SetIntPoint(&ip);
    result += ip.weight * transformation->Weight();
  }
  return result;
}

double C1FaceMeasure(mfem::ParMesh& mesh, const int face, const int order)
{
  mfem::FaceElementTransformations* transformations =
      mesh.GetFaceElementTransformations(face);
  if (transformations == nullptr || transformations->Face == nullptr)
  {
    return 0.0;
  }
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(
      mesh.GetFaceGeometry(face), order);
  double result = 0.0;
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = rule.IntPoint(point);
    transformations->Face->SetIntPoint(&ip);
    result += ip.weight * transformations->Face->Weight();
  }
  return result;
}

struct C1State final
{
  std::vector<double> old_interior;
  std::vector<double> old_true_trace;
  std::vector<double> candidate;
  std::vector<std::vector<double>> old_trace_by_element;
  std::vector<IndependentElementRestriction> restriction;
};

bool C1MaterializeState(
    const sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    C1State& output, IndependentC1Evidence& evidence)
{
  output = {};
  const mfem::ParMesh* mesh = foundation.mesh();
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  const auto* trace_collection = foundation.trace_collection();
  if (mesh == nullptr || trace_collection == nullptr ||
      spaces.sigma.address == nullptr || spaces.trace_carrier.address == nullptr ||
      mesh->GetNE() != c1_elements || mesh->GetNumFaces() != c1_faces ||
      spaces.trace_carrier.local_true_size != c1_true ||
      spaces.trace_carrier.global_true_size != c1_true)
  {
    evidence.failures.push_back("C1 MFEM fixture ownership");
    return false;
  }
  output.old_interior.assign(c1_global_interior, 0.0);
  output.old_true_trace.assign(
      c1_true, std::numeric_limits<double>::quiet_NaN());
  output.candidate.assign(c1_full, 0.0);
  output.old_trace_by_element.assign(
      c1_elements, std::vector<double>(c1_nt, 0.0));
  output.restriction.resize(c1_elements);
  for (int element = 0; element < c1_elements; ++element)
  {
    const mfem::FiniteElement* finite_element =
        spaces.sigma.address->GetFE(element);
    if (finite_element == nullptr || finite_element->GetDof() != c1_ns ||
        finite_element->GetNodes().GetNPoints() != c1_ns)
    {
      evidence.failures.push_back("C1 independent element metadata");
      return false;
    }
    IndependentC0Evidence restriction_evidence;
    if (!BuildIndependentRestriction(
            element, c1_polynomial_degree, *mesh, *trace_collection,
            *spaces.trace_carrier.address,
            output.restriction[static_cast<std::size_t>(element)],
            restriction_evidence))
    {
      evidence.failures.push_back("C1 independent element restriction");
      for (const std::string& failure : restriction_evidence.failures)
      {
        evidence.failures.push_back("C1 restriction: " + failure);
      }
      return false;
    }
    const int ex = element % c1_elements_per_direction;
    const int ey = element / c1_elements_per_direction;
    const double parity = ((ex + ey) % 2 == 0) ? 1.0 : -1.0;
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    for (int scalar = 0; scalar < c1_ns; ++scalar)
    {
      const double xi = 2.0 * nodes.IntPoint(scalar).x - 1.0;
      const double eta = 2.0 * nodes.IntPoint(scalar).y - 1.0;
      const int base = element * c1_ni;
      output.old_interior[base + scalar] =
          1.0 + 0.008 * parity * xi + 0.004 * eta;
      output.old_interior[base + 4 + scalar] = 0.09 + 0.006 * eta;
      output.old_interior[base + 8 + scalar] = -0.04 + 0.004 * xi;
      output.old_interior[base + 12 + scalar] =
          2.505 + 0.001 * parity * xi;
      output.old_interior[base + 16 + scalar] =
          0.018 + 0.003 * xi - 0.001 * eta;
      output.old_interior[base + 20 + scalar] = 0.002 + 0.001 * eta;
      output.old_interior[base + 24 + scalar] = -0.001 + 0.0005 * xi;
      output.candidate[base + scalar] =
          1.0 + 0.010 * parity * xi + 0.005 * eta;
      output.candidate[base + 4 + scalar] = 0.10 + 0.010 * eta;
      output.candidate[base + 8 + scalar] = -0.05 + 0.005 * xi;
      output.candidate[base + 12 + scalar] =
          2.510 + 0.002 * parity * xi;
      output.candidate[base + 16 + scalar] =
          0.020 + 0.004 * xi - 0.002 * eta;
      output.candidate[base + 20 + scalar] = 0.003 + 0.001 * eta;
      output.candidate[base + 24 + scalar] = -0.002 + 0.001 * xi;
    }
  }
  std::vector<unsigned char> assigned(c1_true, 0);
  for (int face = 0; face < mesh->GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* finite_element =
        spaces.trace_carrier.address->GetFaceElement(face);
    mfem::Array<int> face_vdofs;
    spaces.trace_carrier.address->GetFaceVDofs(face, face_vdofs);
    if (finite_element == nullptr || finite_element->GetDof() != c1_face_nodes ||
        finite_element->GetNodes().GetNPoints() != c1_face_nodes ||
        face_vdofs.Size() != c1_trace_components * c1_face_nodes)
    {
      evidence.failures.push_back("C1 independent trace metadata");
      return false;
    }
    for (int component = 0; component < c1_trace_components; ++component)
      for (int scalar = 0; scalar < c1_face_nodes; ++scalar)
      {
        int local = -1;
        int sign = 0;
        if (!DecodeSignedDof(
                face_vdofs[component * c1_face_nodes + scalar], local,
                sign) ||
            local >= spaces.trace_carrier.local_vector_size)
        {
          evidence.failures.push_back("C1 independent trace VDof");
          return false;
        }
        const int local_true =
            spaces.trace_carrier.address->GetLocalTDofNumber(local);
        if (local_true < 0 || local_true >= c1_true)
        {
          evidence.failures.push_back("C1 independent trace true DOF");
          return false;
        }
        const double s =
            2.0 * finite_element->GetNodes().IntPoint(scalar).x - 1.0;
        double old_value = 0.0;
        double value = 0.0;
        if (component == 0)
        {
          old_value = 1.0 + 0.001 * s;
          value = 1.0 + 0.002 * s;
        }
        else if (component == 1)
        {
          old_value = 0.09 + 0.002 * s;
          value = 0.095 + 0.003 * s;
        }
        else if (component == 2)
        {
          old_value = -0.04 + 0.001 * s;
          value = -0.045 + 0.002 * s;
        }
        else if (component == 3)
        {
          old_value = 2.505 + 0.0005 * s;
          value = 2.508 + 0.001 * s;
        }
        else
        {
          old_value = 0.018 + 0.001 * s;
          value = 0.020 + 0.003 * s;
        }
        const double old_true_value = sign * old_value;
        const double true_value = sign * value;
        if (assigned[static_cast<std::size_t>(local_true)] != 0 &&
            (output.old_true_trace[static_cast<std::size_t>(local_true)] !=
                 old_true_value ||
             output.candidate[c1_global_interior + local_true] != true_value))
        {
          evidence.failures.push_back("C1 repeated true trace assignment");
          return false;
        }
        output.old_true_trace[static_cast<std::size_t>(local_true)] =
            old_true_value;
        output.candidate[c1_global_interior + local_true] = true_value;
        assigned[static_cast<std::size_t>(local_true)] = 1;
      }
  }
  if (!std::all_of(assigned.begin(), assigned.end(),
                   [](const unsigned char value) { return value == 1; }))
  {
    evidence.failures.push_back("C1 complete true trace assignment");
    return false;
  }
  std::vector<int> true_incidence(c1_true, 0);
  int restriction_rows = 0;
  int nonidentity_permutation_witnesses = 0;
  for (int element = 0; element < c1_elements; ++element)
  {
    const IndependentElementRestriction& map =
        output.restriction[static_cast<std::size_t>(element)];
    if (map.element != element || map.entries.size() != c1_nt)
    {
      evidence.failures.push_back("C1 independent P-map extent");
      return false;
    }
    for (int row = 0; row < c1_nt; ++row)
    {
      const IndependentRestrictionEntry& entry =
          map.entries[static_cast<std::size_t>(row)];
      if (entry.local_row != row || entry.local_true_dof < 0 ||
          entry.local_true_dof >= c1_true)
      {
        evidence.failures.push_back("C1 independent P-map row");
        return false;
      }
      ++true_incidence[static_cast<std::size_t>(entry.local_true_dof)];
      ++restriction_rows;
      if (entry.oriented_face_dof != entry.canonical_face_dof)
      {
        ++nonidentity_permutation_witnesses;
      }
      output.old_trace_by_element[static_cast<std::size_t>(element)]
                                 [static_cast<std::size_t>(row)] =
          entry.sign * output.old_true_trace[entry.local_true_dof];
    }
  }
  if (restriction_rows != c1_restriction_rows ||
      !std::all_of(true_incidence.begin(), true_incidence.end(),
                   [](const int count) { return count == 2; }))
  {
    evidence.failures.push_back("C1 independent periodic incidence");
    return false;
  }
  if (nonidentity_permutation_witnesses <= 0)
  {
    evidence.failures.push_back(
        "C1 independent nonidentity face-node permutation witness");
    return false;
  }
  return true;
}

template <typename Scalar>
bool C1ElementResidual(
    mfem::ParMesh& mesh, const mfem::ParFiniteElementSpace& volume_space,
    const mfem::ParFiniteElementSpace& trace_space,
    const mfem::DG_Interface_FECollection& trace_collection,
    const IndependentElementRestriction& restriction,
    const std::vector<double>& old, const std::vector<double>& old_trace,
    const std::vector<Scalar>& current, const std::vector<Scalar>& trace,
    const int quadrature_order,
    std::vector<Scalar>& local_residual,
    std::vector<Scalar>& trace_residual)
{
  local_residual.assign(c1_ni, Scalar(0.0));
  trace_residual.assign(c1_nt, Scalar(0.0));
  const int element = restriction.element;
  const mfem::FiniteElement* finite_element = volume_space.GetFE(element);
  mfem::ElementTransformation* transformation =
      mesh.GetElementTransformation(element);
  if (finite_element == nullptr || transformation == nullptr ||
      finite_element->GetDof() != c1_ns || old.size() != c1_ni ||
      old_trace.size() != c1_nt || current.size() != c1_ni ||
      trace.size() != c1_nt ||
      restriction.faces.size() != c1_faces_per_element ||
      restriction.orientations.size() != c1_faces_per_element)
  {
    return false;
  }
  const auto interior = [](const int component, const int scalar)
  { return component * c1_ns + scalar; };
  const auto trace_index = [](const int component, const int scalar)
  { return component * c1_nf + scalar; };
  const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
      finite_element->GetGeomType(), quadrature_order);
  mfem::Vector shape(c1_ns);
  mfem::DenseMatrix derivative(c1_ns, 2);
  for (int point = 0; point < volume_rule.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = volume_rule.IntPoint(point);
    transformation->SetIntPoint(&ip);
    finite_element->CalcShape(ip, shape);
    finite_element->CalcPhysDShape(*transformation, derivative);
    std::array<Scalar, 4> state{}, old_state{};
    std::array<std::array<Scalar, 2>, 4> state_derivative{};
    for (int component = 0; component < 4; ++component)
    {
      state[component] = C1Evaluate(current, component, c1_ns, shape);
      old_state[component] = 0.0;
      for (int scalar = 0; scalar < c1_ns; ++scalar)
      {
        old_state[component] += old[component * c1_ns + scalar] *
            shape[scalar];
      }
      for (int direction = 0; direction < 2; ++direction)
      {
        state_derivative[component][direction] =
            C1Derivative(current, component, c1_ns, derivative, direction);
      }
    }
    if (!C1Admissible(state) || !C1Admissible(old_state)) { return false; }
    const Scalar sigma = C1Evaluate(current, 4, c1_ns, shape);
    const std::array<Scalar, 2> z{{
        C1Evaluate(current, 5, c1_ns, shape),
        C1Evaluate(current, 6, c1_ns, shape)}};
    const auto flux_x = C1Flux(state, sigma, {{1.0, 0.0}});
    const auto flux_y = C1Flux(state, sigma, {{0.0, 1.0}});
    const Scalar rho = state[0];
    const std::array<std::array<Scalar, 2>, 2> velocity_gradient{{
        {{state_derivative[1][0] / rho -
              state[1] * state_derivative[0][0] / (rho * rho),
          state_derivative[1][1] / rho -
              state[1] * state_derivative[0][1] / (rho * rho)}},
        {{state_derivative[2][0] / rho -
              state[2] * state_derivative[0][0] / (rho * rho),
          state_derivative[2][1] / rho -
              state[2] * state_derivative[0][1] / (rho * rho)}}}};
    const Scalar trace_gradient =
        velocity_gradient[0][0] + velocity_gradient[1][1];
    const Scalar q = velocity_gradient[0][0] * velocity_gradient[0][0] +
        2.0 * velocity_gradient[0][1] * velocity_gradient[1][0] +
        velocity_gradient[1][1] * velocity_gradient[1][1] +
        trace_gradient * trace_gradient;
    const double weight = ip.weight * transformation->Weight();
    for (int row_scalar = 0; row_scalar < c1_ns; ++row_scalar)
    {
      for (int equation = 0; equation < 4; ++equation)
      {
        local_residual[interior(equation, row_scalar)] += weight *
            (shape[row_scalar] * (state[equation] - old_state[equation]) /
                 c1_dt -
             derivative(row_scalar, 0) * flux_x[equation] -
             derivative(row_scalar, 1) * flux_y[equation]);
      }
      for (int direction = 0; direction < 2; ++direction)
      {
        local_residual[interior(4 + direction, row_scalar)] += weight *
            (shape[row_scalar] * rho * z[direction] +
             derivative(row_scalar, direction) * sigma);
      }
      local_residual[interior(6, row_scalar)] += weight *
          (shape[row_scalar] * sigma / rho +
           0.1 * derivative(row_scalar, 0) * z[0] +
           0.1 * derivative(row_scalar, 1) * z[1] -
           0.1 * shape[row_scalar] * q);
    }
  }
  const double element_measure =
      C1ElementMeasure(mesh, element, quadrature_order);
  if (!(element_measure > 0.0)) { return false; }
  for (int local_face = 0; local_face < c1_faces_per_element; ++local_face)
  {
    const int face = restriction.faces[static_cast<std::size_t>(local_face)];
    mfem::FaceElementTransformations* face_transformations =
        mesh.GetFaceElementTransformations(face);
    const mfem::FiniteElement* trace_element = trace_space.GetFaceElement(face);
    if (face_transformations == nullptr ||
        face_transformations->Face == nullptr || trace_element == nullptr ||
        trace_element->GetDof() != c1_face_nodes)
    {
      return false;
    }
    const bool element_one = face_transformations->Elem1No == element;
    if (!element_one && face_transformations->Elem2No != element)
    {
      return false;
    }
    const int* permutation = trace_collection.DofOrderForOrientation(
        mfem::Geometry::SEGMENT,
        restriction.orientations[static_cast<std::size_t>(local_face)]);
    const double face_measure =
        C1FaceMeasure(mesh, face, quadrature_order);
    const double height = element_measure / face_measure;
    if (permutation == nullptr || !(face_measure > 0.0) || !(height > 0.0))
    {
      return false;
    }
    const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
        trace_element->GetGeomType(), quadrature_order);
    mfem::Vector interior_shape(c1_ns), face_shape(c1_face_nodes),
        oriented_shape(c1_face_nodes);
    for (int point = 0; point < face_rule.GetNPoints(); ++point)
    {
      const mfem::IntegrationPoint& ip = face_rule.IntPoint(point);
      face_transformations->SetAllIntPoints(&ip);
      const mfem::IntegrationPoint& element_ip = element_one ?
          face_transformations->GetElement1IntPoint() :
          face_transformations->GetElement2IntPoint();
      finite_element->CalcShape(element_ip, interior_shape);
      trace_element->CalcShape(ip, face_shape);
      for (int scalar = 0; scalar < c1_face_nodes; ++scalar)
      {
        int canonical = -1;
        int sign = 0;
        if (!DecodeSignedDof(permutation[scalar], canonical, sign) ||
            canonical >= c1_face_nodes)
        {
          return false;
        }
        oriented_shape[scalar] = sign * face_shape[canonical];
      }
      mfem::Vector scaled(2);
      mfem::CalcOrtho(face_transformations->Face->Jacobian(), scaled);
      const double norm = scaled.Norml2();
      const double measure = face_transformations->Face->Weight();
      if (!(norm > 0.0) || !(measure > 0.0)) { return false; }
      const double side = element_one ? 1.0 : -1.0;
      const std::array<double, 2> normal{{
          side * scaled[0] / norm, side * scaled[1] / norm}};
      std::array<Scalar, 4> state{}, trace_state{};
      std::array<double, 4> old_state{}, old_trace_state{};
      for (int component = 0; component < 4; ++component)
      {
        state[component] =
            C1Evaluate(current, component, c1_ns, interior_shape);
        old_state[component] = 0.0;
        trace_state[component] = 0.0;
        old_trace_state[component] = 0.0;
        for (int scalar = 0; scalar < c1_face_nodes; ++scalar)
        {
          trace_state[component] += trace[component * c1_nf +
              local_face * c1_face_nodes + scalar] * oriented_shape[scalar];
          old_trace_state[component] += old_trace[component * c1_nf +
              local_face * c1_face_nodes + scalar] * oriented_shape[scalar];
        }
        for (int scalar = 0; scalar < c1_ns; ++scalar)
        {
          old_state[component] += old[component * c1_ns + scalar] *
              interior_shape[scalar];
        }
      }
      if (!C1Admissible(state) || !C1Admissible(old_state) ||
          !C1Admissible(trace_state) || !C1Admissible(old_trace_state))
      {
        return false;
      }
      const Scalar sigma = C1Evaluate(current, 4, c1_ns, interior_shape);
      const std::array<Scalar, 2> z{{
          C1Evaluate(current, 5, c1_ns, interior_shape),
          C1Evaluate(current, 6, c1_ns, interior_shape)}};
      Scalar sigmahat = 0.0;
      for (int scalar = 0; scalar < c1_face_nodes; ++scalar)
      {
        sigmahat += trace[c1_state_components * c1_nf +
                          local_face * c1_face_nodes + scalar] *
            oriented_shape[scalar];
      }
      const auto physical_flux = C1Flux(trace_state, sigmahat, normal);
      std::array<Scalar, 4> numerical_flux{};
      for (int equation = 0; equation < 4; ++equation)
      {
        numerical_flux[equation] = physical_flux[equation] +
            2.5 * (state[equation] - trace_state[equation]);
      }
      const Scalar tau =
          static_cast<double>(c1_face_nodes * c1_face_nodes) /
          (trace_state[0] * height);
      const Scalar auxiliary_flux = z[0] * normal[0] + z[1] * normal[1] -
          tau * (sigma - sigmahat);
      const double weight = ip.weight * measure;
      for (int row_scalar = 0; row_scalar < c1_ns; ++row_scalar)
      {
        for (int equation = 0; equation < 4; ++equation)
        {
          local_residual[interior(equation, row_scalar)] +=
              weight * interior_shape[row_scalar] * numerical_flux[equation];
        }
        for (int direction = 0; direction < 2; ++direction)
        {
          local_residual[interior(4 + direction, row_scalar)] -= weight *
              interior_shape[row_scalar] * normal[direction] * sigmahat;
        }
        local_residual[interior(6, row_scalar)] -=
            0.1 * weight * interior_shape[row_scalar] * auxiliary_flux;
      }
      for (int row_component = 0; row_component < c1_trace_components;
           ++row_component)
        for (int row_scalar = 0; row_scalar < c1_face_nodes; ++row_scalar)
        {
          const int row = trace_index(row_component,
              local_face * c1_face_nodes + row_scalar);
          trace_residual[row] += weight * oriented_shape[row_scalar] *
              (row_component < 4 ? numerical_flux[row_component] :
                                   0.1 * auxiliary_flux);
        }
    }
  }
  return true;
}

struct C1Assembly final
{
  std::vector<sc25::ElementLinearization> elements;
  std::vector<double> residual;
  std::vector<double> jacobian;
};

bool C1AssembleAnalytic(
    const sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const C1State& state, const std::vector<double>& candidate,
    const int quadrature_order, C1Assembly& output)
{
  output = {};
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  const auto* trace_collection = foundation.trace_collection();
  if (mesh == nullptr || spaces.sigma.address == nullptr ||
      spaces.trace_carrier.address == nullptr || trace_collection == nullptr ||
      candidate.size() != c1_full ||
      state.restriction.size() != c1_elements)
  {
    return false;
  }
  output.elements.resize(c1_elements);
  output.residual.assign(c1_full, 0.0);
  output.jacobian.assign(static_cast<std::size_t>(c1_full) * c1_full, 0.0);
  for (int element = 0; element < c1_elements; ++element)
  {
    std::vector<C1Dual> local(c1_ni), trace(c1_nt);
    for (int entry = 0; entry < c1_ni; ++entry)
    {
      local[entry] = C1Dual::Variable(
          candidate[element * c1_ni + entry], entry);
    }
    const IndependentElementRestriction& map =
        state.restriction[static_cast<std::size_t>(element)];
    for (int row = 0; row < c1_nt; ++row)
    {
      const IndependentRestrictionEntry& entry = map.entries[row];
      trace[row] = C1Dual::Variable(
          entry.sign *
              candidate[c1_global_interior + entry.local_true_dof],
          c1_ni + row);
    }
    std::vector<double> old(c1_ni);
    std::copy(state.old_interior.begin() + element * c1_ni,
              state.old_interior.begin() + (element + 1) * c1_ni,
              old.begin());
    std::vector<C1Dual> local_residual, trace_residual;
    if (!C1ElementResidual(
            *mesh, *spaces.sigma.address, *spaces.trace_carrier.address,
            *trace_collection, map, old,
            state.old_trace_by_element[static_cast<std::size_t>(element)],
            local, trace, quadrature_order, local_residual, trace_residual))
    {
      return false;
    }
    sc25::ElementLinearization& block = output.elements[element];
    block.element = element;
    block.local_interior_unknowns = c1_ni;
    block.local_trace_unknowns = c1_nt;
    block.a.assign(static_cast<std::size_t>(c1_ni) * c1_ni, 0.0);
    block.b.assign(static_cast<std::size_t>(c1_ni) * c1_nt, 0.0);
    block.c.assign(static_cast<std::size_t>(c1_nt) * c1_ni, 0.0);
    block.d.assign(static_cast<std::size_t>(c1_nt) * c1_nt, 0.0);
    block.local_residual.resize(c1_ni);
    block.trace_residual.resize(c1_nt);
    for (int row = 0; row < c1_ni; ++row)
    {
      block.local_residual[row] = local_residual[row].value;
      output.residual[element * c1_ni + row] = local_residual[row].value;
      for (int column = 0; column < c1_ni; ++column)
      {
        block.a[static_cast<std::size_t>(row) * c1_ni + column] =
            local_residual[row].derivative[column];
        output.jacobian[
            static_cast<std::size_t>(element * c1_ni + row) * c1_full +
            element * c1_ni + column] =
            local_residual[row].derivative[column];
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        block.b[static_cast<std::size_t>(row) * c1_nt + column] =
            local_residual[row].derivative[c1_ni + column];
        const auto& entry = map.entries[column];
        output.jacobian[
            static_cast<std::size_t>(element * c1_ni + row) * c1_full +
            c1_global_interior + entry.local_true_dof] += entry.sign *
            block.b[static_cast<std::size_t>(row) * c1_nt + column];
      }
    }
    for (int row = 0; row < c1_nt; ++row)
    {
      block.trace_residual[row] = trace_residual[row].value;
      const auto& row_entry = map.entries[row];
      const int true_row =
          c1_global_interior + row_entry.local_true_dof;
      output.residual[true_row] += row_entry.sign * trace_residual[row].value;
      for (int column = 0; column < c1_ni; ++column)
      {
        block.c[static_cast<std::size_t>(row) * c1_ni + column] =
            trace_residual[row].derivative[column];
        output.jacobian[static_cast<std::size_t>(true_row) * c1_full +
                        element * c1_ni + column] += row_entry.sign *
            block.c[static_cast<std::size_t>(row) * c1_ni + column];
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        block.d[static_cast<std::size_t>(row) * c1_nt + column] =
            trace_residual[row].derivative[c1_ni + column];
        const auto& column_entry = map.entries[column];
        output.jacobian[static_cast<std::size_t>(true_row) * c1_full +
                        c1_global_interior +
                            column_entry.local_true_dof] +=
            row_entry.sign * column_entry.sign *
            block.d[static_cast<std::size_t>(row) * c1_nt + column];
      }
    }
  }
  return true;
}

bool C1ResidualOnly(
    const sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const C1State& state, const std::vector<double>& candidate,
    const int quadrature_order, std::vector<double>& residual)
{
  residual.assign(c1_full, 0.0);
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  const auto* trace_collection = foundation.trace_collection();
  if (mesh == nullptr || spaces.sigma.address == nullptr ||
      spaces.trace_carrier.address == nullptr || trace_collection == nullptr ||
      candidate.size() != c1_full ||
      state.restriction.size() != c1_elements)
  {
    return false;
  }
  for (int element = 0; element < c1_elements; ++element)
  {
    std::vector<double> local(c1_ni), trace(c1_nt), old(c1_ni);
    std::copy(candidate.begin() + element * c1_ni,
              candidate.begin() + (element + 1) * c1_ni, local.begin());
    std::copy(state.old_interior.begin() + element * c1_ni,
              state.old_interior.begin() + (element + 1) * c1_ni,
              old.begin());
    const auto& map = state.restriction[element];
    for (int row = 0; row < c1_nt; ++row)
    {
      trace[row] = map.entries[row].sign *
          candidate[c1_global_interior +
                    map.entries[row].local_true_dof];
    }
    std::vector<double> local_residual, trace_residual;
    if (!C1ElementResidual(
            *mesh, *spaces.sigma.address, *spaces.trace_carrier.address,
            *trace_collection, map, old,
            state.old_trace_by_element[static_cast<std::size_t>(element)],
            local, trace, quadrature_order, local_residual, trace_residual))
    {
      return false;
    }
    std::copy(local_residual.begin(), local_residual.end(),
              residual.begin() + element * c1_ni);
    for (int row = 0; row < c1_nt; ++row)
    {
      residual[c1_global_interior + map.entries[row].local_true_dof] +=
          map.entries[row].sign * trace_residual[row];
    }
  }
  return true;
}

struct C1Lu final
{
  int size = 0;
  std::vector<double> values;

  bool Factor(const std::vector<double>& matrix, const int n)
  {
    if (n <= 0 || matrix.size() != static_cast<std::size_t>(n) * n)
    {
      return false;
    }
    size = n;
    values = matrix;
    for (const double value : values)
    {
      if (!std::isfinite(value)) { return false; }
    }
    return true;
  }

  bool Solve(const std::vector<double>& rhs, std::vector<double>& solution) const
  {
    return DirectAugmentedPartialPivotSolve(values, size, rhs, solution);
  }
};

double C1Relative(const std::vector<double>& actual,
                  const std::vector<double>& expected)
{
  if (actual.size() != expected.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  return RelativeL2(actual.data(), expected.data(),
                    static_cast<int>(expected.size()));
}

double C1InfDirectional(const std::vector<double>& left,
                        const std::vector<double>& right)
{
  if (left.size() != right.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  double difference = 0.0;
  double expected = 0.0;
  for (std::size_t entry = 0; entry < left.size(); ++entry)
  {
    difference = std::max(difference, std::abs(left[entry] - right[entry]));
    expected = std::max(expected, std::abs(right[entry]));
  }
  return difference / std::max(1.0, expected);
}

} // namespace

bool RunIndependentC1Oracle(
    sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    IndependentC1Evidence& evidence)
{
  evidence = {};
  C1State state;
  if (!C1MaterializeState(foundation, state, evidence)) { return false; }
  sc25::CoupledOperatorFixtureSnapshot production;
  if (!foundation.RunCoupledOperatorFixture(production))
  {
    evidence.failures.push_back("production C1 fixture execution");
    return false;
  }
  evidence.full_dimension = production.full_unknowns;
  if (production.quadrature_order != c1_quadrature_order ||
      production.repeat_quadrature_order != c1_repeat_quadrature_order ||
      production.local_interior_unknowns != c1_ni ||
      production.local_trace_unknowns != c1_nt ||
      production.true_trace_unknowns != c1_true ||
      production.full_unknowns != c1_full ||
      production.old_interior.size() != c1_global_interior ||
      production.old_true_trace.size() != c1_true ||
      production.candidate_interior.size() != c1_global_interior ||
      production.candidate_true_trace.size() != c1_true)
  {
    evidence.failures.push_back("production C1 dimension audit");
    return false;
  }
  const sc25::CoupledTraceSolverEvidence& solver = production.solver;
  if (solver.restart_dimension != 50 || solver.maximum_iterations != 200 ||
      solver.relative_tolerance != 1.0e-12 ||
      solver.absolute_tolerance != 1.0e-14 || solver.ilu_type != 0 ||
      solver.ilu_level_of_fill != 0 || solver.ilu_local_reordering != 1 ||
      solver.setup_status != 0 || solver.solve_status != 0 ||
      solver.convergence_query_status != 0 ||
      solver.converged != 1 || solver.iterations < 0 ||
      solver.iterations > 200 ||
      !std::isfinite(solver.unpreconditioned_residual_norm) ||
      !std::isfinite(solver.right_hand_side_norm) ||
      solver.unpreconditioned_residual_norm >
          solver.absolute_tolerance +
              solver.relative_tolerance * solver.right_hand_side_norm ||
      !std::isfinite(production.maximum_local_application_residual))
  {
    evidence.failures.push_back("production C1 solver contract");
  }
  std::vector<double> expected_candidate_interior(
      state.candidate.begin(),
      state.candidate.begin() + c1_global_interior);
  std::vector<double> expected_true_trace(
      state.candidate.begin() + c1_global_interior,
      state.candidate.end());
  if (C1Relative(production.old_interior, state.old_interior) != 0.0 ||
      C1Relative(production.old_true_trace, state.old_true_trace) != 0.0 ||
      C1Relative(production.candidate_interior,
                 expected_candidate_interior) != 0.0 ||
      C1Relative(production.candidate_true_trace, expected_true_trace) != 0.0)
  {
    evidence.failures.push_back("exact C1 state materialization");
  }

  C1Assembly oracle, repeat;
  if (!C1AssembleAnalytic(foundation, state, state.candidate,
                          c1_quadrature_order, oracle) ||
      !C1AssembleAnalytic(foundation, state, state.candidate,
                          c1_repeat_quadrature_order, repeat))
  {
    evidence.failures.push_back("independent C1 analytic assembly");
    return false;
  }
  evidence.residual_relative_l2 =
      C1Relative(production.full_residual, oracle.residual);
  evidence.jacobian_relative_l2 =
      C1Relative(production.full_jacobian, oracle.jacobian);
  if (evidence.residual_relative_l2 > 2.0e-11)
  {
    evidence.failures.push_back("C1 full residual comparison");
  }
  if (evidence.jacobian_relative_l2 > 2.0e-11)
  {
    evidence.failures.push_back("C1 full Jacobian comparison");
  }
  if (production.elements.size() != c1_elements ||
      oracle.elements.size() != c1_elements)
  {
    evidence.failures.push_back("C1 local block extent");
    return false;
  }
  for (int element = 0; element < c1_elements; ++element)
  {
    const auto& actual = production.elements[element];
    const auto& expected = oracle.elements[element];
    evidence.maximum_local_block_relative_l2 = std::max({
        evidence.maximum_local_block_relative_l2,
        C1Relative(actual.local_residual, expected.local_residual),
        C1Relative(actual.trace_residual, expected.trace_residual),
        C1Relative(actual.a, expected.a), C1Relative(actual.b, expected.b),
        C1Relative(actual.c, expected.c), C1Relative(actual.d, expected.d)});
  }
  if (evidence.maximum_local_block_relative_l2 > 2.0e-11)
  {
    evidence.failures.push_back("C1 local A/B/C/D comparison");
  }
  evidence.quadrature_repeat_difference = std::max(
      C1Relative(oracle.residual, repeat.residual),
      C1Relative(oracle.jacobian, repeat.jacobian));
  if (evidence.quadrature_repeat_difference > 1.0e-11 ||
      production.residual_quadrature_repeat_relative_l2 > 1.0e-11 ||
      production.jacobian_quadrature_repeat_relative_l2 > 1.0e-11)
  {
    evidence.failures.push_back("C1 quadrature repeat");
  }

  std::vector<C1Lu> local_lu(c1_elements);
  std::vector<double> oracle_schur(
      static_cast<std::size_t>(c1_elements) * c1_nt * c1_nt, 0.0);
  std::vector<double> oracle_condensed_rhs(c1_restriction_rows, 0.0);
  for (int element = 0; element < c1_elements; ++element)
  {
    const auto& block = oracle.elements[element];
    if (!local_lu[element].Factor(block.a, c1_ni))
    {
      evidence.failures.push_back("C1 independent local factor");
      return false;
    }
    std::vector<double> inverse_residual;
    if (!local_lu[element].Solve(block.local_residual, inverse_residual))
    {
      evidence.failures.push_back("C1 independent local residual solve");
      return false;
    }
    std::vector<double> inverse_b(c1_ni * c1_nt, 0.0);
    for (int column = 0; column < c1_nt; ++column)
    {
      std::vector<double> rhs(c1_ni), solution;
      for (int row = 0; row < c1_ni; ++row)
      {
        rhs[row] = block.b[static_cast<std::size_t>(row) * c1_nt + column];
      }
      if (!local_lu[element].Solve(rhs, solution)) { return false; }
      for (int row = 0; row < c1_ni; ++row)
      {
        inverse_b[static_cast<std::size_t>(row) * c1_nt + column] =
            solution[row];
      }
    }
    for (int row = 0; row < c1_nt; ++row)
    {
      oracle_condensed_rhs[element * c1_nt + row] =
          -block.trace_residual[row];
      for (int inner = 0; inner < c1_ni; ++inner)
      {
        oracle_condensed_rhs[element * c1_nt + row] +=
            block.c[static_cast<std::size_t>(row) * c1_ni + inner] *
            inverse_residual[inner];
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        double value = block.d[static_cast<std::size_t>(row) * c1_nt + column];
        for (int inner = 0; inner < c1_ni; ++inner)
        {
          value -= block.c[static_cast<std::size_t>(row) * c1_ni + inner] *
              inverse_b[static_cast<std::size_t>(inner) * c1_nt + column];
        }
        oracle_schur[(static_cast<std::size_t>(element) * c1_nt + row) *
                         c1_nt + column] = value;
      }
    }
  }
  std::vector<double> production_schur, production_condensed_rhs;
  for (const auto& block : production.condensed)
  {
    production_schur.insert(production_schur.end(), block.schur.begin(),
                            block.schur.end());
    production_condensed_rhs.insert(production_condensed_rhs.end(),
                                    block.right_hand_side.begin(),
                                    block.right_hand_side.end());
  }
  evidence.schur_relative_l2 =
      C1Relative(production_schur, oracle_schur);
  evidence.condensed_rhs_relative_l2 =
      C1Relative(production_condensed_rhs, oracle_condensed_rhs);
  if (evidence.schur_relative_l2 > 2.0e-11 ||
      evidence.condensed_rhs_relative_l2 > 2.0e-11)
  {
    evidence.failures.push_back("C1 local condensation");
  }

  std::vector<double> oracle_global_schur(
      static_cast<std::size_t>(c1_true) * c1_true, 0.0);
  std::vector<double> oracle_global_rhs(c1_true, 0.0);
  for (int element = 0; element < c1_elements; ++element)
    for (int row = 0; row < c1_nt; ++row)
    {
      const IndependentRestrictionEntry& row_entry =
          state.restriction[static_cast<std::size_t>(element)].entries[
              static_cast<std::size_t>(row)];
      oracle_global_rhs[row_entry.local_true_dof] += row_entry.sign *
          oracle_condensed_rhs[element * c1_nt + row];
      for (int column = 0; column < c1_nt; ++column)
      {
        const IndependentRestrictionEntry& column_entry =
            state.restriction[static_cast<std::size_t>(element)].entries[
                static_cast<std::size_t>(column)];
        oracle_global_schur[
            static_cast<std::size_t>(row_entry.local_true_dof) * c1_true +
            column_entry.local_true_dof] +=
            row_entry.sign * column_entry.sign *
            oracle_schur[(static_cast<std::size_t>(element) * c1_nt + row) *
                             c1_nt + column];
      }
    }
  const mfem::HypreParMatrix* assembled_matrix =
      foundation.assembled_true_trace_matrix();
  const mfem::Vector* assembled_rhs =
      foundation.assembled_true_trace_right_hand_side();
  if (assembled_matrix == nullptr || assembled_rhs == nullptr ||
      assembled_matrix->Height() != c1_true ||
      assembled_matrix->Width() != c1_true ||
      assembled_rhs->Size() != c1_true)
  {
    evidence.failures.push_back("C1 assembled true system extent");
    return false;
  }
  std::vector<double> production_global_schur(
      static_cast<std::size_t>(c1_true) * c1_true, 0.0);
  mfem::Vector basis(c1_true), action(c1_true);
  for (int column = 0; column < c1_true; ++column)
  {
    basis = 0.0;
    basis[column] = 1.0;
    assembled_matrix->Mult(basis, action);
    const double* action_values = action.HostRead();
    for (int row = 0; row < c1_true; ++row)
    {
      production_global_schur[static_cast<std::size_t>(row) * c1_true +
                              column] = action_values[row];
    }
  }
  std::vector<double> production_global_rhs(
      assembled_rhs->HostRead(), assembled_rhs->HostRead() + c1_true);
  evidence.assembled_schur_relative_l2 =
      C1Relative(production_global_schur, oracle_global_schur);
  evidence.assembled_rhs_relative_l2 =
      C1Relative(production_global_rhs, oracle_global_rhs);
  if (evidence.assembled_schur_relative_l2 > 2.0e-11 ||
      evidence.assembled_rhs_relative_l2 > 2.0e-11)
  {
    evidence.failures.push_back("C1 assembled global Schur/RHS");
  }
  C1Lu condensed_lu;
  std::vector<double> condensed_increment;
  if (!condensed_lu.Factor(oracle_global_schur, c1_true) ||
      !condensed_lu.Solve(oracle_global_rhs, condensed_increment))
  {
    evidence.failures.push_back("C1 independent condensed solve");
    return false;
  }

  std::vector<double> dense_rhs(c1_full);
  for (int row = 0; row < c1_full; ++row)
  {
    dense_rhs[row] = -oracle.residual[row];
  }
  C1Lu full_lu;
  std::vector<double> dense_increment;
  if (!full_lu.Factor(oracle.jacobian, c1_full) ||
      !full_lu.Solve(dense_rhs, dense_increment))
  {
    evidence.failures.push_back("C1 independent full dense solve");
    return false;
  }
  std::vector<double> dense_true(
                                 dense_increment.begin() + c1_global_interior,
                                 dense_increment.end());
  evidence.condensed_solve_relative_l2 = std::max(
      C1Relative(condensed_increment, dense_true),
      C1Relative(production.true_trace_increment, condensed_increment));
  evidence.true_trace_increment_relative_l2 =
      C1Relative(production.true_trace_increment, dense_true);
  evidence.full_increment_relative_l2 =
      C1Relative(production.full_increment, dense_increment);
  std::vector<double> expected_restricted(c1_restriction_rows, 0.0);
  for (int element = 0; element < c1_elements; ++element)
    for (int row = 0; row < c1_nt; ++row)
    {
      const auto& entry = state.restriction[element].entries[row];
      expected_restricted[element * c1_nt + row] =
          entry.sign * dense_true[entry.local_true_dof];
    }
  evidence.restricted_increment_relative_l2 = C1Relative(
      production.element_trace_increment, expected_restricted);
  std::vector<double> dense_recovered(dense_increment.begin(),
      dense_increment.begin() + c1_global_interior);
  evidence.recovered_increment_relative_l2 = C1Relative(
      production.recovered_interior_increment, dense_recovered);
  if (std::max({evidence.condensed_solve_relative_l2,
                evidence.true_trace_increment_relative_l2,
                evidence.full_increment_relative_l2,
                evidence.restricted_increment_relative_l2,
                evidence.recovered_increment_relative_l2}) > 2.0e-11)
  {
    evidence.failures.push_back("C1 solve/restriction/recovery");
  }

  double y_norm_inf = 0.0;
  for (const double value : state.candidate)
  {
    y_norm_inf = std::max(y_norm_inf, std::abs(value));
  }
  for (int direction_index = 0; direction_index <= c1_full;
       ++direction_index)
  {
    std::vector<double> direction(c1_full, 0.0);
    if (direction_index < c1_full)
    {
      direction[direction_index] = 1.0;
    }
    else
    {
      for (int entry = 0; entry < c1_full; ++entry)
      {
        direction[entry] = (entry % 2 == 0 ? 1.0 : -1.0) *
            static_cast<double>(entry + 1) / c1_full;
      }
    }
    double direction_norm_inf = 0.0;
    for (const double value : direction)
    {
      direction_norm_inf = std::max(direction_norm_inf, std::abs(value));
    }
    const double epsilon = std::ldexp(1.0, -20) *
        std::max(1.0, y_norm_inf) / std::max(1.0, direction_norm_inf);
    std::vector<double> plus = state.candidate;
    std::vector<double> minus = state.candidate;
    for (int entry = 0; entry < c1_full; ++entry)
    {
      plus[entry] += epsilon * direction[entry];
      minus[entry] -= epsilon * direction[entry];
    }
    std::vector<double> plus_residual, minus_residual;
    if (!C1ResidualOnly(foundation, state, plus, c1_quadrature_order,
                        plus_residual) ||
        !C1ResidualOnly(foundation, state, minus, c1_quadrature_order,
                        minus_residual))
    {
      evidence.failures.push_back("C1 directional residual-only evaluation");
      return false;
    }
    std::vector<double> finite_difference(c1_full, 0.0);
    std::vector<double> jacobian_action(c1_full, 0.0);
    for (int row = 0; row < c1_full; ++row)
    {
      finite_difference[row] =
          (plus_residual[row] - minus_residual[row]) / (2.0 * epsilon);
      for (int column = 0; column < c1_full; ++column)
      {
        jacobian_action[row] += oracle.jacobian[
            static_cast<std::size_t>(row) * c1_full + column] *
            direction[column];
      }
    }
    const double discrepancy =
        C1InfDirectional(jacobian_action, finite_difference);
    evidence.maximum_directional_discrepancy =
        std::max(evidence.maximum_directional_discrepancy, discrepancy);
    if (discrepancy > 2.0e-6)
    {
      evidence.failures.push_back(
          "C1 directional Jacobian " + std::to_string(direction_index));
    }
    if (direction_index < c1_full)
    {
      ++evidence.checked_coordinate_directions;
    }
  }
  if (evidence.checked_coordinate_directions != c1_full)
  {
    evidence.failures.push_back("C1 coordinate-direction census");
  }
  return evidence.failures.empty();
}

namespace
{

using TrajectoryState = sc25::PersistentTrajectoryState;
using TrajectoryHashes = sc25::TrajectoryStateHashes;

void TrajectoryRequire(const bool condition, const std::string& message,
                       IndependentTrajectoryEvidence& evidence)
{
  if (!condition) { evidence.failures.push_back(message); }
}

template <typename Array>
dg::hdg::runtime::StateHash HashOneArray(const Array& values) noexcept
{
  return {dg::hdg::runtime::ComputeSha256(
      values.data(), values.size() * sizeof(double))};
}

template <typename... Arrays>
dg::hdg::runtime::StateHash HashArraySequence(
    const Arrays&... arrays) noexcept
{
  dg::hdg::runtime::Sha256 hash;
  const bool updated =
      (hash.Update(arrays.data(), arrays.size() * sizeof(double)) && ...);
  dg::hdg::runtime::Sha256Digest digest{};
  if (!updated || !hash.Finalize(digest)) { return {}; }
  return {digest};
}

TrajectoryHashes IndependentHashes(const TrajectoryState& state) noexcept
{
  return {HashOneArray(state.u),
          HashOneArray(state.sigma),
          HashOneArray(state.z),
          HashOneArray(state.uhat),
          HashOneArray(state.sigmahat),
          HashArraySequence(state.u, state.sigma, state.z),
          HashArraySequence(state.uhat, state.sigmahat)};
}

bool HashesEqual(const TrajectoryHashes& left,
                 const TrajectoryHashes& right) noexcept
{
  return left.u == right.u && left.sigma == right.sigma &&
      left.z == right.z && left.uhat == right.uhat &&
      left.sigmahat == right.sigmahat &&
      left.interior == right.interior && left.trace == right.trace;
}

bool ReuseSnapshotsEqual(const sc25::TrajectoryReuseSnapshot& left,
                         const sc25::TrajectoryReuseSnapshot& right) noexcept
{
  return left.foundation_handle == right.foundation_handle &&
      left.topology_handle == right.topology_handle &&
      left.carrier_map_handle == right.carrier_map_handle &&
      left.contract_handle == right.contract_handle &&
      left.assembly_state_handle == right.assembly_state_handle &&
      left.assembly_handle == right.assembly_handle &&
      left.trial_assembly_handle == right.trial_assembly_handle &&
      left.factor_handle == right.factor_handle &&
      left.workspace_handle == right.workspace_handle &&
      left.quadrature_workspace_handle == right.quadrature_workspace_handle &&
      left.local_pattern_handle == right.local_pattern_handle &&
      left.true_matrix_handle == right.true_matrix_handle &&
      left.true_rhs_handle == right.true_rhs_handle &&
      left.true_solution_handle == right.true_solution_handle &&
      left.ilu_handle == right.ilu_handle &&
      left.gmres_handle == right.gmres_handle &&
      left.true_residual_handle == right.true_residual_handle &&
      left.element_to_true_csr_handle == right.element_to_true_csr_handle &&
      left.owned_boundary_identity_hash ==
          right.owned_boundary_identity_hash &&
      left.topology_generation == right.topology_generation &&
      left.storage_generation == right.storage_generation &&
      left.factor_generation == right.factor_generation &&
      left.workspace_generation == right.workspace_generation &&
      left.local_pattern_generation == right.local_pattern_generation &&
      left.solver_generation == right.solver_generation &&
      left.w8_owned_boundary_object_count ==
          right.w8_owned_boundary_object_count &&
      left.w8_owned_boundary_buffer_count ==
          right.w8_owned_boundary_buffer_count &&
      !left.opaque_mfem_hypre_observed &&
      !right.opaque_mfem_hypre_observed;
}

bool FiniteState(const TrajectoryState& state) noexcept
{
  const auto finite = [](const auto& values)
  {
    return std::all_of(values.begin(), values.end(),
                       [](const double value) { return std::isfinite(value); });
  };
  return finite(state.u) && finite(state.sigma) && finite(state.z) &&
      finite(state.uhat) && finite(state.sigmahat);
}

bool PrimitiveAdmissible(
    const std::array<double, sc25::state_components>& conservative,
    double& density, double& pressure, double& speed_margin) noexcept
{
  density = conservative[0];
  if (!std::isfinite(density) || density < 1.0e-8) { return false; }
  const double inverse_density = 1.0 / density;
  const double velocity_x = conservative[1] * inverse_density;
  const double velocity_y = conservative[2] * inverse_density;
  pressure = (sc25::gamma - 1.0) *
      (conservative[3] -
       0.5 * (conservative[1] * conservative[1] +
              conservative[2] * conservative[2]) * inverse_density);
  if (!std::isfinite(pressure) || pressure < 1.0e-8) { return false; }
  const double sound = std::sqrt(sc25::gamma * pressure * inverse_density);
  speed_margin = 2.5 - std::hypot(velocity_x, velocity_y) - sound;
  return std::isfinite(speed_margin) && speed_margin > 0.0;
}

bool IndependentVolumeChecks(
    sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const TrajectoryState& state,
    std::array<double, sc25::state_components>& integrals,
    double& minimum_density, double& minimum_pressure,
    double& minimum_speed_margin, const int quadrature_order) noexcept
{
  integrals.fill(0.0);
  minimum_density = std::numeric_limits<double>::infinity();
  minimum_pressure = std::numeric_limits<double>::infinity();
  minimum_speed_margin = std::numeric_limits<double>::infinity();
  const mfem::ParFiniteElementSpace* space = foundation.spaces().u.address;
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  if (!space || !mesh || mesh->GetNE() != sc25::persistent_elements)
  {
    return false;
  }
  try
  {
    for (int element = 0; element < sc25::persistent_elements; ++element)
    {
      const mfem::FiniteElement* finite_element = space->GetFE(element);
      mfem::ElementTransformation* transformation =
          mesh->GetElementTransformation(element);
      if (!finite_element || !transformation ||
          finite_element->GetDof() != sc25::persistent_scalar_volume_nodes)
      {
        return false;
      }
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), quadrature_order);
      mfem::Vector shape(sc25::persistent_scalar_volume_nodes);
      const std::size_t base = static_cast<std::size_t>(element) *
          sc25::state_components * sc25::persistent_scalar_volume_nodes;
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& ip = rule.IntPoint(point);
        transformation->SetIntPoint(&ip);
        finite_element->CalcShape(ip, shape);
        const double weight = ip.weight * transformation->Weight();
        std::array<double, sc25::state_components> conservative{};
        for (int component = 0; component < sc25::state_components;
             ++component)
          for (int scalar = 0;
               scalar < sc25::persistent_scalar_volume_nodes; ++scalar)
          {
            conservative[component] += shape[scalar] * state.u[
                base + static_cast<std::size_t>(component) *
                           sc25::persistent_scalar_volume_nodes + scalar];
          }
        double density = 0.0;
        double pressure = 0.0;
        double speed_margin = 0.0;
        if (!PrimitiveAdmissible(conservative, density, pressure,
                                 speed_margin) ||
            !std::isfinite(weight) || !(weight > 0.0))
        {
          return false;
        }
        minimum_density = std::min(minimum_density, density);
        minimum_pressure = std::min(minimum_pressure, pressure);
        minimum_speed_margin = std::min(minimum_speed_margin, speed_margin);
        for (int component = 0; component < sc25::state_components;
             ++component)
        {
          integrals[component] += weight * conservative[component];
        }
      }
    }
    return std::all_of(integrals.begin(), integrals.end(),
                       [](const double value) { return std::isfinite(value); });
  }
  catch (...)
  {
    return false;
  }
}

bool IndependentTraceNodeChecks(const TrajectoryState& state,
                                double& minimum_density,
                                double& minimum_pressure,
                                double& minimum_speed_margin) noexcept
{
  for (int scalar = 0; scalar < sc25::persistent_scalar_true_trace_nodes;
       ++scalar)
  {
    std::array<double, sc25::state_components> conservative{};
    for (int component = 0; component < sc25::state_components; ++component)
    {
      conservative[component] = state.uhat[
          static_cast<std::size_t>(component) *
              sc25::persistent_scalar_true_trace_nodes + scalar];
    }
    double density = 0.0;
    double pressure = 0.0;
    double speed_margin = 0.0;
    if (!PrimitiveAdmissible(conservative, density, pressure, speed_margin))
    {
      return false;
    }
    minimum_density = std::min(minimum_density, density);
    minimum_pressure = std::min(minimum_pressure, pressure);
    minimum_speed_margin = std::min(minimum_speed_margin, speed_margin);
  }
  return true;
}

std::array<double, sc25::state_components> IndependentInitialState(
    const double x, const double y) noexcept
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double density = 1.0 + 0.05 * std::sin(2.0 * pi * x) *
      std::sin(2.0 * pi * y);
  const double velocity_x = 0.1 * std::sin(2.0 * pi * x);
  const double velocity_y = 0.1 * std::sin(2.0 * pi * y);
  return {{density, density * velocity_x, density * velocity_y,
           1.0 / (frozen_gamma - 1.0) +
               0.5 * density *
                   (velocity_x * velocity_x + velocity_y * velocity_y)}};
}

bool VerifyIndependentInitialL2Projection(
    sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const TrajectoryState& state,
    const int quadrature_order, double& maximum_volume_moment,
    double& maximum_trace_moment) noexcept
{
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  const mfem::ParFiniteElementSpace* volume_space = spaces.u.address;
  const mfem::ParFiniteElementSpace* trace_space = spaces.uhat.address;
  if (!mesh || !volume_space || !trace_space) { return false; }
  maximum_volume_moment = 0.0;
  maximum_trace_moment = 0.0;
  try
  {
    constexpr int ns = sc25::persistent_scalar_volume_nodes;
    constexpr int nf = sc25::coupled_fixture_scalar_face_nodes;
    for (int element = 0; element < sc25::persistent_elements; ++element)
    {
      const mfem::FiniteElement* finite_element =
          volume_space->GetFE(element);
      mfem::ElementTransformation* transformation =
          mesh->GetElementTransformation(element);
      if (!finite_element || !transformation ||
          finite_element->GetDof() != ns)
      {
        return false;
      }
      mfem::Vector shape(ns), point(sc25::dimension);
      std::array<std::array<long double, ns>, sc25::state_components>
          moments{};
      std::array<std::array<long double, ns>, sc25::state_components>
          scales{};
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), quadrature_order);
      const std::size_t base = static_cast<std::size_t>(element) *
          sc25::state_components * ns;
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& ip = rule.IntPoint(q);
        transformation->SetIntPoint(&ip);
        finite_element->CalcShape(ip, shape);
        transformation->Transform(ip, point);
        const auto exact = IndependentInitialState(point[0], point[1]);
        const double weight = ip.weight * transformation->Weight();
        for (int component = 0; component < sc25::state_components;
             ++component)
        {
          double projected = 0.0;
          for (int scalar = 0; scalar < ns; ++scalar)
          {
            projected += shape[scalar] * state.u[
                base + static_cast<std::size_t>(component) * ns + scalar];
          }
          for (int row = 0; row < ns; ++row)
          {
            const long double term = static_cast<long double>(weight) *
                shape[row] * (projected - exact[component]);
            moments[static_cast<std::size_t>(component)]
                   [static_cast<std::size_t>(row)] += term;
            scales[static_cast<std::size_t>(component)]
                  [static_cast<std::size_t>(row)] += std::abs(
                      static_cast<long double>(weight) * shape[row] *
                      (projected + exact[component]));
          }
        }
      }
      for (int component = 0; component < sc25::state_components;
           ++component)
        for (int row = 0; row < ns; ++row)
        {
          const long double moment =
              moments[static_cast<std::size_t>(component)]
                     [static_cast<std::size_t>(row)];
          const long double scale =
              scales[static_cast<std::size_t>(component)]
                    [static_cast<std::size_t>(row)];
          maximum_volume_moment = std::max(
              maximum_volume_moment,
              static_cast<double>(std::abs(moment) /
                                  std::max(1.0L, scale)));
        }
    }

    for (int face = 0; face < mesh->GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* finite_element =
          trace_space->GetFaceElement(face);
      mfem::ElementTransformation* transformation =
          mesh->GetFaceTransformation(face);
      mfem::Array<int> signed_dofs;
      trace_space->GetFaceVDofs(face, signed_dofs);
      if (!finite_element || !transformation ||
          finite_element->GetDof() != nf ||
          signed_dofs.Size() != sc25::state_components * nf)
      {
        return false;
      }
      mfem::Vector shape(nf), point(sc25::dimension);
      std::array<std::array<long double, nf>, sc25::state_components>
          moments{};
      std::array<std::array<long double, nf>, sc25::state_components>
          scales{};
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), quadrature_order);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& ip = rule.IntPoint(q);
        transformation->SetIntPoint(&ip);
        finite_element->CalcShape(ip, shape);
        transformation->Transform(ip, point);
        const auto exact = IndependentInitialState(point[0], point[1]);
        const double weight = ip.weight * transformation->Weight();
        for (int component = 0; component < sc25::state_components;
             ++component)
        {
          double projected = 0.0;
          for (int scalar = 0; scalar < nf; ++scalar)
          {
            int dof = -1;
            int sign = 0;
            if (!DecodeSignedDof(signed_dofs[component * nf + scalar],
                                 dof, sign))
            {
              return false;
            }
            const int true_dof = trace_space->GetLocalTDofNumber(dof);
            if (true_dof < 0 ||
                true_dof >= sc25::persistent_uhat_unknowns)
            {
              return false;
            }
            projected += shape[scalar] * sign *
                state.uhat[static_cast<std::size_t>(true_dof)];
          }
          for (int row = 0; row < nf; ++row)
          {
            const long double term = static_cast<long double>(weight) *
                shape[row] * (projected - exact[component]);
            moments[static_cast<std::size_t>(component)]
                   [static_cast<std::size_t>(row)] += term;
            scales[static_cast<std::size_t>(component)]
                  [static_cast<std::size_t>(row)] += std::abs(
                      static_cast<long double>(weight) * shape[row] *
                      (projected + exact[component]));
          }
        }
      }
      for (int component = 0; component < sc25::state_components;
           ++component)
        for (int row = 0; row < nf; ++row)
        {
          const long double moment =
              moments[static_cast<std::size_t>(component)]
                     [static_cast<std::size_t>(row)];
          const long double scale =
              scales[static_cast<std::size_t>(component)]
                    [static_cast<std::size_t>(row)];
          maximum_trace_moment = std::max(
              maximum_trace_moment,
              static_cast<double>(std::abs(moment) /
                                  std::max(1.0L, scale)));
        }
    }
  }
  catch (...)
  {
    return false;
  }
  return std::isfinite(maximum_volume_moment) &&
      std::isfinite(maximum_trace_moment);
}

TrajectoryState ApplyIndependentDirection(
    const TrajectoryState& entry, const TrajectoryState& direction,
    const double theta) noexcept
{
  TrajectoryState trial = entry;
  const auto add = [theta](auto& output, const auto& increment)
  {
    for (std::size_t i = 0; i < output.size(); ++i)
    {
      output[i] += theta * increment[i];
    }
  };
  add(trial.u, direction.u);
  add(trial.sigma, direction.sigma);
  add(trial.z, direction.z);
  add(trial.uhat, direction.uhat);
  add(trial.sigmahat, direction.sigmahat);
  return trial;
}

struct N8OracleContext final
{
  std::vector<IndependentElementRestriction> restriction;
};

double N8TrueValue(const TrajectoryState& state, const int true_dof)
{
  return true_dof < sc25::persistent_uhat_unknowns ?
      state.uhat[static_cast<std::size_t>(true_dof)] :
      state.sigmahat[static_cast<std::size_t>(
          true_dof - sc25::persistent_uhat_unknowns)];
}

void N8PackElement(const TrajectoryState& state, const int element,
                   std::vector<double>& local)
{
  local.assign(c1_ni, 0.0);
  const std::size_t u_base = static_cast<std::size_t>(element) *
      sc25::state_components * c1_ns;
  const std::size_t sigma_base = static_cast<std::size_t>(element) * c1_ns;
  const std::size_t z_base = static_cast<std::size_t>(element) * 2 * c1_ns;
  std::copy_n(state.u.begin() + u_base, sc25::state_components * c1_ns,
              local.begin());
  std::copy_n(state.sigma.begin() + sigma_base, c1_ns,
              local.begin() + 4 * c1_ns);
  std::copy_n(state.z.begin() + z_base, 2 * c1_ns,
              local.begin() + 5 * c1_ns);
}

void N8PackTrace(const TrajectoryState& state,
                 const IndependentElementRestriction& map,
                 std::vector<double>& trace)
{
  trace.assign(c1_nt, 0.0);
  for (int row = 0; row < c1_nt; ++row)
  {
    const IndependentRestrictionEntry& entry =
        map.entries[static_cast<std::size_t>(row)];
    trace[static_cast<std::size_t>(row)] =
        entry.sign * N8TrueValue(state, entry.local_true_dof);
  }
}

bool BuildN8OracleContext(
    const sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    N8OracleContext& context, IndependentTrajectoryEvidence& evidence)
{
  context = {};
  const mfem::ParMesh* mesh = foundation.mesh();
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  const mfem::DG_Interface_FECollection* trace_collection =
      foundation.trace_collection();
  if (!mesh || !spaces.trace_carrier.address || !trace_collection ||
      mesh->GetNE() != sc25::persistent_elements ||
      mesh->GetNumFaces() != sc25::persistent_unique_faces ||
      spaces.trace_carrier.local_true_size !=
          sc25::persistent_true_trace_unknowns)
  {
    evidence.failures.push_back("N8 independent ownership");
    return false;
  }
  context.restriction.resize(sc25::persistent_elements);
  std::vector<int> incidence(sc25::persistent_true_trace_unknowns, 0);
  for (int element = 0; element < sc25::persistent_elements; ++element)
  {
    IndependentC0Evidence restriction_evidence;
    if (!BuildIndependentRestriction(
            element, 1, *mesh, *trace_collection,
            *spaces.trace_carrier.address,
            context.restriction[static_cast<std::size_t>(element)],
            restriction_evidence))
    {
      evidence.failures.push_back("N8 independent restriction " +
                                  std::to_string(element));
      return false;
    }
    const IndependentElementRestriction& map =
        context.restriction[static_cast<std::size_t>(element)];
    if (map.entries.size() != c1_nt) { return false; }
    for (const IndependentRestrictionEntry& entry : map.entries)
    {
      if (entry.local_true_dof < 0 ||
          entry.local_true_dof >= sc25::persistent_true_trace_unknowns)
      {
        return false;
      }
      ++incidence[static_cast<std::size_t>(entry.local_true_dof)];
    }
  }
  if (!std::all_of(incidence.begin(), incidence.end(),
                   [](const int value) { return value == 2; }))
  {
    evidence.failures.push_back("N8 independent periodic incidence");
    return false;
  }
  return true;
}

bool EvaluateIndependentAuxiliaryResidualImpl(
    sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const TrajectoryState& state, const int quadrature_order,
    sc25::AuxiliaryResidualMetrics& metrics)
{
  metrics = {};
  IndependentTrajectoryEvidence context_evidence;
  N8OracleContext context;
  if (!BuildN8OracleContext(foundation, context, context_evidence))
  {
    return false;
  }
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  const mfem::DG_Interface_FECollection* trace_collection =
      foundation.trace_collection();
  if (!mesh || !spaces.sigma.address || !spaces.trace_carrier.address ||
      !trace_collection)
  {
    return false;
  }
  constexpr int rz_rows = n8_elements * 2 * c1_ns;
  constexpr int rsigma_rows = n8_elements * c1_ns;
  std::array<std::vector<double>, 3> rz_terms;
  std::array<std::vector<double>, 4> rsigma_terms;
  for (auto& term : rz_terms) { term.assign(rz_rows, 0.0); }
  for (auto& term : rsigma_terms) { term.assign(rsigma_rows, 0.0); }
  std::vector<double> g_sigma(sc25::persistent_sigmahat_unknowns, 0.0);
  double g_sigma_term_norm_sum = 0.0;
  try
  {
    for (int element = 0; element < n8_elements; ++element)
    {
      const IndependentElementRestriction& restriction =
          context.restriction[static_cast<std::size_t>(element)];
      std::vector<double> current, trace;
      N8PackElement(state, element, current);
      N8PackTrace(state, restriction, trace);
      const mfem::FiniteElement* finite_element =
          spaces.sigma.address->GetFE(element);
      mfem::ElementTransformation* transformation =
          mesh->GetElementTransformation(element);
      if (!finite_element || !transformation ||
          finite_element->GetDof() != c1_ns)
      {
        return false;
      }
      mfem::Vector shape(c1_ns);
      mfem::DenseMatrix derivative(c1_ns, c1_dimension);
      const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), quadrature_order);
      for (int point = 0; point < volume_rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& ip = volume_rule.IntPoint(point);
        transformation->SetIntPoint(&ip);
        finite_element->CalcShape(ip, shape);
        finite_element->CalcPhysDShape(*transformation, derivative);
        std::array<double, c1_state_components> conservative{};
        std::array<std::array<double, c1_dimension>, c1_state_components>
            conservative_derivative{};
        for (int component = 0; component < c1_state_components; ++component)
        {
          conservative[component] =
              C1Evaluate(current, component, c1_ns, shape);
          for (int direction = 0; direction < c1_dimension; ++direction)
          {
            conservative_derivative[component][direction] =
                C1Derivative(current, component, c1_ns, derivative,
                             direction);
          }
        }
        if (!C1Admissible(conservative)) { return false; }
        const double sigma = C1Evaluate(current, 4, c1_ns, shape);
        const std::array<double, c1_dimension> z{{
            C1Evaluate(current, 5, c1_ns, shape),
            C1Evaluate(current, 6, c1_ns, shape)}};
        const double rho = conservative[0];
        std::array<std::array<double, c1_dimension>, c1_dimension> gradient{};
        for (int direction = 0; direction < c1_dimension; ++direction)
        {
          gradient[0][direction] =
              conservative_derivative[1][direction] / rho -
              conservative[1] * conservative_derivative[0][direction] /
                  (rho * rho);
          gradient[1][direction] =
              conservative_derivative[2][direction] / rho -
              conservative[2] * conservative_derivative[0][direction] /
                  (rho * rho);
        }
        const double trace_gradient = gradient[0][0] + gradient[1][1];
        const double q = gradient[0][0] * gradient[0][0] +
            2.0 * gradient[0][1] * gradient[1][0] +
            gradient[1][1] * gradient[1][1] +
            trace_gradient * trace_gradient;
        const double weight = ip.weight * transformation->Weight();
        if (!std::isfinite(weight) || !(weight > 0.0)) { return false; }
        for (int scalar = 0; scalar < c1_ns; ++scalar)
        {
          for (int direction = 0; direction < c1_dimension; ++direction)
          {
            const std::size_t row =
                static_cast<std::size_t>(element) * 2 * c1_ns +
                direction * c1_ns + scalar;
            rz_terms[0][row] += weight * shape[scalar] * rho * z[direction];
            rz_terms[1][row] +=
                weight * derivative(scalar, direction) * sigma;
          }
          const std::size_t row =
              static_cast<std::size_t>(element) * c1_ns + scalar;
          rsigma_terms[0][row] += weight * shape[scalar] * sigma / rho;
          rsigma_terms[1][row] += 0.1 * weight *
              (derivative(scalar, 0) * z[0] +
               derivative(scalar, 1) * z[1]);
          rsigma_terms[2][row] -=
              0.1 * weight * shape[scalar] * q;
        }
      }

      const double element_measure =
          C1ElementMeasure(*mesh, element, quadrature_order);
      for (int local_face = 0; local_face < c1_faces_per_element;
           ++local_face)
      {
        const int face =
            restriction.faces[static_cast<std::size_t>(local_face)];
        mfem::FaceElementTransformations* face_transformations =
            mesh->GetFaceElementTransformations(face);
        const mfem::FiniteElement* trace_element =
            spaces.trace_carrier.address->GetFaceElement(face);
        if (!face_transformations || !face_transformations->Face ||
            !trace_element || trace_element->GetDof() != c1_face_nodes)
        {
          return false;
        }
        const bool element_one = face_transformations->Elem1No == element;
        if (!element_one && face_transformations->Elem2No != element)
        {
          return false;
        }
        const int* permutation = trace_collection->DofOrderForOrientation(
            mfem::Geometry::SEGMENT,
            restriction.orientations[static_cast<std::size_t>(local_face)]);
        const double face_measure =
            C1FaceMeasure(*mesh, face, quadrature_order);
        const double height = element_measure / face_measure;
        if (!permutation || !(element_measure > 0.0) ||
            !(face_measure > 0.0) || !(height > 0.0))
        {
          return false;
        }
        mfem::Vector interior_shape(c1_ns), face_shape(c1_face_nodes),
            oriented_shape(c1_face_nodes), scaled(c1_dimension);
        std::array<double, c1_face_nodes> side_g_sigma{};
        const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
            trace_element->GetGeomType(), quadrature_order);
        for (int point = 0; point < face_rule.GetNPoints(); ++point)
        {
          const mfem::IntegrationPoint& ip = face_rule.IntPoint(point);
          face_transformations->SetAllIntPoints(&ip);
          const mfem::IntegrationPoint& element_ip = element_one ?
              face_transformations->GetElement1IntPoint() :
              face_transformations->GetElement2IntPoint();
          finite_element->CalcShape(element_ip, interior_shape);
          trace_element->CalcShape(ip, face_shape);
          for (int scalar = 0; scalar < c1_face_nodes; ++scalar)
          {
            int canonical = -1;
            int sign = 0;
            if (!DecodeSignedDof(permutation[scalar], canonical, sign) ||
                canonical >= c1_face_nodes)
            {
              return false;
            }
            oriented_shape[scalar] = sign * face_shape[canonical];
          }
          mfem::CalcOrtho(face_transformations->Face->Jacobian(), scaled);
          const double normal_norm = scaled.Norml2();
          const double measure = face_transformations->Face->Weight();
          if (!(normal_norm > 0.0) || !(measure > 0.0)) { return false; }
          const double side = element_one ? 1.0 : -1.0;
          const std::array<double, c1_dimension> normal{{
              side * scaled[0] / normal_norm,
              side * scaled[1] / normal_norm}};
          std::array<double, c1_state_components> conservative{},
              trace_state{};
          for (int component = 0; component < c1_state_components;
               ++component)
          {
            conservative[component] =
                C1Evaluate(current, component, c1_ns, interior_shape);
            for (int scalar = 0; scalar < c1_face_nodes; ++scalar)
            {
              trace_state[component] += trace[
                  static_cast<std::size_t>(component) * c1_nf +
                  local_face * c1_face_nodes + scalar] *
                  oriented_shape[scalar];
            }
          }
          if (!C1Admissible(conservative) || !C1Admissible(trace_state))
          {
            return false;
          }
          const double sigma =
              C1Evaluate(current, 4, c1_ns, interior_shape);
          const std::array<double, c1_dimension> z{{
              C1Evaluate(current, 5, c1_ns, interior_shape),
              C1Evaluate(current, 6, c1_ns, interior_shape)}};
          double sigmahat = 0.0;
          for (int scalar = 0; scalar < c1_face_nodes; ++scalar)
          {
            sigmahat += trace[static_cast<std::size_t>(4) * c1_nf +
                              local_face * c1_face_nodes + scalar] *
                oriented_shape[scalar];
          }
          const double tau = 4.0 / (trace_state[0] * height);
          const double auxiliary_flux = z[0] * normal[0] +
              z[1] * normal[1] - tau * (sigma - sigmahat);
          const double weight = ip.weight * measure;
          if (!std::isfinite(tau) || !(tau > 0.0) ||
              !std::isfinite(auxiliary_flux) || !std::isfinite(weight))
          {
            return false;
          }
          for (int scalar = 0; scalar < c1_ns; ++scalar)
          {
            for (int direction = 0; direction < c1_dimension; ++direction)
            {
              const std::size_t row =
                  static_cast<std::size_t>(element) * 2 * c1_ns +
                  direction * c1_ns + scalar;
              rz_terms[2][row] -= weight * interior_shape[scalar] *
                  normal[direction] * sigmahat;
            }
            const std::size_t row =
                static_cast<std::size_t>(element) * c1_ns + scalar;
            rsigma_terms[3][row] -= 0.1 * weight *
                interior_shape[scalar] * auxiliary_flux;
          }
          for (int scalar = 0; scalar < c1_face_nodes; ++scalar)
          {
            side_g_sigma[static_cast<std::size_t>(scalar)] +=
                0.1 * weight * oriented_shape[scalar] * auxiliary_flux;
          }
        }
        double side_square = 0.0;
        for (int scalar = 0; scalar < c1_face_nodes; ++scalar)
        {
          const IndependentRestrictionEntry& entry = restriction.entries[
              static_cast<std::size_t>(32 +
                  local_face * c1_face_nodes + scalar)];
          const int true_dof = entry.local_true_dof -
              sc25::persistent_uhat_unknowns;
          if (true_dof < 0 ||
              true_dof >= sc25::persistent_sigmahat_unknowns)
          {
            return false;
          }
          const double contribution = entry.sign *
              side_g_sigma[static_cast<std::size_t>(scalar)];
          g_sigma[static_cast<std::size_t>(true_dof)] += contribution;
          side_square += contribution * contribution;
        }
        g_sigma_term_norm_sum += std::sqrt(side_square);
      }
    }

    const auto norm = [](const std::vector<double>& values)
    {
      long double square = 0.0L;
      for (const double value : values)
      {
        square += static_cast<long double>(value) * value;
      }
      return std::sqrt(static_cast<double>(square));
    };
    std::vector<double> r_z(rz_rows, 0.0), r_sigma(rsigma_rows, 0.0);
    for (std::size_t row = 0; row < r_z.size(); ++row)
      for (const auto& term : rz_terms) { r_z[row] += term[row]; }
    for (std::size_t row = 0; row < r_sigma.size(); ++row)
      for (const auto& term : rsigma_terms) { r_sigma[row] += term[row]; }
    metrics.r_z_norm = norm(r_z);
    metrics.r_sigma_norm = norm(r_sigma);
    metrics.g_sigma_norm = norm(g_sigma);
    for (const auto& term : rz_terms) { metrics.r_z_scale += norm(term); }
    for (const auto& term : rsigma_terms)
    {
      metrics.r_sigma_scale += norm(term);
    }
    metrics.r_z_scale = std::max(1.0, metrics.r_z_scale);
    metrics.r_sigma_scale = std::max(1.0, metrics.r_sigma_scale);
    metrics.g_sigma_scale = std::max(1.0, g_sigma_term_norm_sum);
    metrics.normalized_maximum = std::max({
        metrics.r_z_norm / metrics.r_z_scale,
        metrics.r_sigma_norm / metrics.r_sigma_scale,
        metrics.g_sigma_norm / metrics.g_sigma_scale});
    return std::isfinite(metrics.r_z_norm) &&
        std::isfinite(metrics.r_z_scale) &&
        std::isfinite(metrics.r_sigma_norm) &&
        std::isfinite(metrics.r_sigma_scale) &&
        std::isfinite(metrics.g_sigma_norm) &&
        std::isfinite(metrics.g_sigma_scale) &&
        std::isfinite(metrics.normalized_maximum);
  }
  catch (...)
  {
    metrics = {};
    return false;
  }
}

struct N8Assembly final
{
  std::vector<sc25::ElementLinearization> elements;
  std::vector<double> residual;
};

bool N8ResidualOnly(
    const sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const N8OracleContext& context, const TrajectoryState& old_state,
    const TrajectoryState& candidate, const int quadrature_order,
    std::vector<double>& residual)
{
  residual.assign(sc25::persistent_full_unknowns, 0.0);
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  const mfem::DG_Interface_FECollection* trace_collection =
      foundation.trace_collection();
  if (!mesh || !spaces.sigma.address || !spaces.trace_carrier.address ||
      !trace_collection ||
      context.restriction.size() != sc25::persistent_elements)
  {
    return false;
  }
  for (int element = 0; element < sc25::persistent_elements; ++element)
  {
    std::vector<double> old, old_trace, current, trace;
    N8PackElement(old_state, element, old);
    N8PackTrace(old_state,
                context.restriction[static_cast<std::size_t>(element)],
                old_trace);
    N8PackElement(candidate, element, current);
    const IndependentElementRestriction& map =
        context.restriction[static_cast<std::size_t>(element)];
    N8PackTrace(candidate, map, trace);
    std::vector<double> local_residual, trace_residual;
    if (!C1ElementResidual(
            *mesh, *spaces.sigma.address, *spaces.trace_carrier.address,
            *trace_collection, map, old, old_trace, current, trace,
            quadrature_order, local_residual, trace_residual))
    {
      return false;
    }
    std::copy(local_residual.begin(), local_residual.end(),
              residual.begin() + element * c1_ni);
    for (int row = 0; row < c1_nt; ++row)
    {
      const IndependentRestrictionEntry& entry =
          map.entries[static_cast<std::size_t>(row)];
      residual[sc25::persistent_interior_unknowns + entry.local_true_dof] +=
          entry.sign * trace_residual[static_cast<std::size_t>(row)];
    }
  }
  return std::all_of(residual.begin(), residual.end(),
                     [](const double value) { return std::isfinite(value); });
}

bool N8AssembleAnalytic(
    const sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const N8OracleContext& context, const TrajectoryState& old_state,
    const TrajectoryState& candidate, const int quadrature_order,
    N8Assembly& output)
{
  output = {};
  output.elements.resize(sc25::persistent_elements);
  output.residual.assign(sc25::persistent_full_unknowns, 0.0);
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  const mfem::DG_Interface_FECollection* trace_collection =
      foundation.trace_collection();
  if (!mesh || !spaces.sigma.address || !spaces.trace_carrier.address ||
      !trace_collection ||
      context.restriction.size() != sc25::persistent_elements)
  {
    return false;
  }
  for (int element = 0; element < sc25::persistent_elements; ++element)
  {
    const IndependentElementRestriction& map =
        context.restriction[static_cast<std::size_t>(element)];
    std::vector<double> old, old_trace, current_values, trace_values;
    N8PackElement(old_state, element, old);
    N8PackTrace(old_state, map, old_trace);
    N8PackElement(candidate, element, current_values);
    N8PackTrace(candidate, map, trace_values);
    std::vector<C1Dual> current(c1_ni), trace(c1_nt);
    for (int entry = 0; entry < c1_ni; ++entry)
    {
      current[static_cast<std::size_t>(entry)] = C1Dual::Variable(
          current_values[static_cast<std::size_t>(entry)], entry);
    }
    for (int row = 0; row < c1_nt; ++row)
    {
      trace[static_cast<std::size_t>(row)] = C1Dual::Variable(
          trace_values[static_cast<std::size_t>(row)], c1_ni + row);
    }
    std::vector<C1Dual> local_residual, trace_residual;
    if (!C1ElementResidual(
            *mesh, *spaces.sigma.address, *spaces.trace_carrier.address,
            *trace_collection, map, old, old_trace, current, trace,
            quadrature_order, local_residual, trace_residual))
    {
      return false;
    }
    sc25::ElementLinearization& block =
        output.elements[static_cast<std::size_t>(element)];
    block.element = element;
    block.local_interior_unknowns = c1_ni;
    block.local_trace_unknowns = c1_nt;
    block.a.assign(static_cast<std::size_t>(c1_ni) * c1_ni, 0.0);
    block.b.assign(static_cast<std::size_t>(c1_ni) * c1_nt, 0.0);
    block.c.assign(static_cast<std::size_t>(c1_nt) * c1_ni, 0.0);
    block.d.assign(static_cast<std::size_t>(c1_nt) * c1_nt, 0.0);
    block.local_residual.resize(c1_ni);
    block.trace_residual.resize(c1_nt);
    for (int row = 0; row < c1_ni; ++row)
    {
      block.local_residual[static_cast<std::size_t>(row)] =
          local_residual[static_cast<std::size_t>(row)].value;
      output.residual[static_cast<std::size_t>(element) * c1_ni + row] =
          block.local_residual[static_cast<std::size_t>(row)];
      for (int column = 0; column < c1_ni; ++column)
      {
        block.a[static_cast<std::size_t>(row) * c1_ni + column] =
            local_residual[static_cast<std::size_t>(row)]
                .derivative[static_cast<std::size_t>(column)];
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        block.b[static_cast<std::size_t>(row) * c1_nt + column] =
            local_residual[static_cast<std::size_t>(row)]
                .derivative[static_cast<std::size_t>(c1_ni + column)];
      }
    }
    for (int row = 0; row < c1_nt; ++row)
    {
      block.trace_residual[static_cast<std::size_t>(row)] =
          trace_residual[static_cast<std::size_t>(row)].value;
      const IndependentRestrictionEntry& row_entry =
          map.entries[static_cast<std::size_t>(row)];
      output.residual[sc25::persistent_interior_unknowns +
                      row_entry.local_true_dof] +=
          row_entry.sign * block.trace_residual[static_cast<std::size_t>(row)];
      for (int column = 0; column < c1_ni; ++column)
      {
        block.c[static_cast<std::size_t>(row) * c1_ni + column] =
            trace_residual[static_cast<std::size_t>(row)]
                .derivative[static_cast<std::size_t>(column)];
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        block.d[static_cast<std::size_t>(row) * c1_nt + column] =
            trace_residual[static_cast<std::size_t>(row)]
                .derivative[static_cast<std::size_t>(c1_ni + column)];
      }
    }
  }
  return true;
}

double N8Norm(const std::vector<double>& values)
{
  long double square = 0.0L;
  for (const double value : values)
  {
    square += static_cast<long double>(value) * value;
  }
  return std::sqrt(static_cast<double>(square));
}

double N8AssemblyRelative(const N8Assembly& left,
                          const N8Assembly& right)
{
  if (left.elements.size() != right.elements.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  double maximum = C1Relative(left.residual, right.residual);
  for (std::size_t element = 0; element < left.elements.size(); ++element)
  {
    const sc25::ElementLinearization& a = left.elements[element];
    const sc25::ElementLinearization& b = right.elements[element];
    maximum = std::max({maximum, C1Relative(a.a, b.a),
                        C1Relative(a.b, b.b), C1Relative(a.c, b.c),
                        C1Relative(a.d, b.d),
                        C1Relative(a.local_residual, b.local_residual),
                        C1Relative(a.trace_residual, b.trace_residual)});
  }
  return maximum;
}

bool VerifyN8Direction(
    const N8OracleContext& context, const N8Assembly& assembly,
    const TrajectoryState& direction,
    IndependentTrajectoryEvidence& evidence, const std::string& label)
{
  std::vector<double> global_action(sc25::persistent_full_unknowns, 0.0);
  std::vector<double> global_condensed_action(
      sc25::persistent_true_trace_unknowns, 0.0);
  std::vector<double> global_condensed_rhs(
      sc25::persistent_true_trace_unknowns, 0.0);
  double maximum_recovery = 0.0;
  for (int element = 0; element < sc25::persistent_elements; ++element)
  {
    const sc25::ElementLinearization& block =
        assembly.elements[static_cast<std::size_t>(element)];
    const IndependentElementRestriction& map =
        context.restriction[static_cast<std::size_t>(element)];
    std::vector<double> local_direction, trace_direction;
    N8PackElement(direction, element, local_direction);
    N8PackTrace(direction, map, trace_direction);
    for (int row = 0; row < c1_ni; ++row)
    {
      double value = block.local_residual[static_cast<std::size_t>(row)];
      for (int column = 0; column < c1_ni; ++column)
      {
        value += block.a[static_cast<std::size_t>(row) * c1_ni + column] *
            local_direction[static_cast<std::size_t>(column)];
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        value += block.b[static_cast<std::size_t>(row) * c1_nt + column] *
            trace_direction[static_cast<std::size_t>(column)];
      }
      global_action[static_cast<std::size_t>(element) * c1_ni + row] = value;
    }
    for (int row = 0; row < c1_nt; ++row)
    {
      double value = block.trace_residual[static_cast<std::size_t>(row)];
      for (int column = 0; column < c1_ni; ++column)
      {
        value += block.c[static_cast<std::size_t>(row) * c1_ni + column] *
            local_direction[static_cast<std::size_t>(column)];
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        value += block.d[static_cast<std::size_t>(row) * c1_nt + column] *
            trace_direction[static_cast<std::size_t>(column)];
      }
      const IndependentRestrictionEntry& entry =
          map.entries[static_cast<std::size_t>(row)];
      global_action[sc25::persistent_interior_unknowns +
                    entry.local_true_dof] += entry.sign * value;
    }

    C1Lu factor;
    std::vector<double> inverse_residual;
    if (!factor.Factor(block.a, c1_ni) ||
        !factor.Solve(block.local_residual, inverse_residual))
    {
      evidence.failures.push_back(label + " independent local factor");
      return false;
    }
    std::vector<double> inverse_b(
        static_cast<std::size_t>(c1_ni) * c1_nt, 0.0);
    for (int column = 0; column < c1_nt; ++column)
    {
      std::vector<double> rhs(c1_ni), solution;
      for (int row = 0; row < c1_ni; ++row)
      {
        rhs[static_cast<std::size_t>(row)] =
            block.b[static_cast<std::size_t>(row) * c1_nt + column];
      }
      if (!factor.Solve(rhs, solution)) { return false; }
      for (int row = 0; row < c1_ni; ++row)
      {
        inverse_b[static_cast<std::size_t>(row) * c1_nt + column] =
            solution[static_cast<std::size_t>(row)];
      }
    }
    std::vector<double> recovery_rhs = block.local_residual;
    for (int row = 0; row < c1_ni; ++row)
      for (int column = 0; column < c1_nt; ++column)
      {
        recovery_rhs[static_cast<std::size_t>(row)] +=
            block.b[static_cast<std::size_t>(row) * c1_nt + column] *
            trace_direction[static_cast<std::size_t>(column)];
      }
    std::vector<double> recovered;
    if (!factor.Solve(recovery_rhs, recovered)) { return false; }
    for (int row = 0; row < c1_ni; ++row)
    {
      maximum_recovery = std::max(
          maximum_recovery,
          std::abs(local_direction[static_cast<std::size_t>(row)] +
                   recovered[static_cast<std::size_t>(row)]));
    }
    for (int row = 0; row < c1_nt; ++row)
    {
      double rhs_value = -block.trace_residual[static_cast<std::size_t>(row)];
      for (int inner = 0; inner < c1_ni; ++inner)
      {
        rhs_value += block.c[static_cast<std::size_t>(row) * c1_ni + inner] *
            inverse_residual[static_cast<std::size_t>(inner)];
      }
      double action_value = -rhs_value;
      for (int column = 0; column < c1_nt; ++column)
      {
        double schur =
            block.d[static_cast<std::size_t>(row) * c1_nt + column];
        for (int inner = 0; inner < c1_ni; ++inner)
        {
          schur -= block.c[static_cast<std::size_t>(row) * c1_ni + inner] *
              inverse_b[static_cast<std::size_t>(inner) * c1_nt + column];
        }
        action_value += schur *
            trace_direction[static_cast<std::size_t>(column)];
      }
      const IndependentRestrictionEntry& entry =
          map.entries[static_cast<std::size_t>(row)];
      global_condensed_action[static_cast<std::size_t>(entry.local_true_dof)] +=
          entry.sign * action_value;
      global_condensed_rhs[static_cast<std::size_t>(entry.local_true_dof)] +=
          entry.sign * rhs_value;
    }
  }
  const double action_relative = N8Norm(global_action) /
      std::max(1.0, N8Norm(assembly.residual));
  const double condensed_relative = N8Norm(global_condensed_action) /
      std::max(1.0, N8Norm(global_condensed_rhs));
  const double independent_condensed_residual =
      N8Norm(global_condensed_action);
  const double independent_condensed_rhs = N8Norm(global_condensed_rhs);
  evidence.maximum_jacobian_action_relative_l2 = std::max(
      evidence.maximum_jacobian_action_relative_l2, action_relative);
  evidence.maximum_condensed_action_relative_l2 = std::max(
      evidence.maximum_condensed_action_relative_l2, condensed_relative);
  evidence.maximum_recovery_relative_l2 = std::max(
      evidence.maximum_recovery_relative_l2, maximum_recovery);
  TrajectoryRequire(action_relative <= 2.0e-11,
                    label + " independent full Jacobian action", evidence);
  TrajectoryRequire(condensed_relative <= 2.0e-11,
                    label + " independent condensed action", evidence);
  TrajectoryRequire(
      independent_condensed_residual <=
          1.0e-14 + 1.0e-12 * independent_condensed_rhs,
      label + " independent true trace residual bound", evidence);
  TrajectoryRequire(maximum_recovery <= 2.0e-11,
                    label + " independent local recovery", evidence);
  return true;
}

} // namespace

bool EvaluateIndependentAuxiliaryResidualOracle(
    sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const sc25::PersistentTrajectoryState& state,
    const int quadrature_order,
    sc25::AuxiliaryResidualMetrics& metrics)
{
  return EvaluateIndependentAuxiliaryResidualImpl(
      foundation, state, quadrature_order, metrics);
}

bool RunIndependentTrajectoryOracle(
    sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    IndependentTrajectoryEvidence& evidence)
{
  evidence = {};
  evidence.minimum_density = std::numeric_limits<double>::infinity();
  evidence.minimum_pressure = std::numeric_limits<double>::infinity();
  evidence.minimum_speed_margin = std::numeric_limits<double>::infinity();
  const sc25::PersistentTrajectoryEvidence& run =
      foundation.trajectory_evidence();
  TrajectoryRequire(
      run.status == sc25::TrajectoryStatus::HdgSc25Accepted ||
          run.status == sc25::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry,
      "trajectory terminal status", evidence);
  TrajectoryRequire(run.accepted_steps == sc25::persistent_required_steps,
                    "trajectory accepted-step count", evidence);
  const std::uint64_t checkpoint_offset = run.checkpoint_step_offset;
  const int expected_continuation_steps = checkpoint_offset <= n8_steps ?
      n8_steps - static_cast<int>(checkpoint_offset) : -1;
  TrajectoryRequire(
      checkpoint_offset <= static_cast<std::uint64_t>(n8_steps) &&
          run.step_evidence_count == expected_continuation_steps,
      "trajectory honest continuation step-evidence count", evidence);
  AttemptStepMapping attempt_step_mapping;
  attempt_step_mapping.step_count = run.step_evidence_count;
  attempt_step_mapping.attempt_count = run.attempt_count;
  if (run.step_evidence_count >= 0 &&
      run.step_evidence_count <= n8_steps)
  {
    for (int index = 0; index < run.step_evidence_count; ++index)
    {
      attempt_step_mapping.step_numbers[static_cast<std::size_t>(index)] =
          run.steps[static_cast<std::size_t>(index)].step;
      attempt_step_mapping.step_attempt_indices[
          static_cast<std::size_t>(index)] =
          run.steps[static_cast<std::size_t>(index)].attempt_index;
    }
  }
  if (run.attempt_count >= 0 && run.attempt_count <= n8_attempt_capacity)
  {
    for (int index = 0; index < run.attempt_count; ++index)
    {
      attempt_step_mapping.attempt_status[static_cast<std::size_t>(index)] =
          run.attempts[static_cast<std::size_t>(index)].status;
      attempt_step_mapping.attempt_target_steps[
          static_cast<std::size_t>(index)] =
          run.attempts[static_cast<std::size_t>(index)].target_step;
    }
  }
  const bool valid_attempt_step_bijection =
      ValidAcceptedAttemptStepBijection(attempt_step_mapping,
                                        checkpoint_offset);
  TrajectoryRequire(valid_attempt_step_bijection,
                    "trajectory accepted step/attempt bijection", evidence);
  if (valid_attempt_step_bijection && attempt_step_mapping.step_count > 0)
  {
    AttemptStepMapping failed_substitution = attempt_step_mapping;
    int failed_attempt_index = -1;
    for (int index = 0; index < failed_substitution.attempt_count; ++index)
    {
      const int target = failed_substitution.attempt_target_steps[
          static_cast<std::size_t>(index)];
      if (!AcceptedAttemptStatus(failed_substitution.attempt_status[
              static_cast<std::size_t>(index)]) &&
          target > static_cast<int>(checkpoint_offset) &&
          target <= n8_steps)
      {
        failed_attempt_index = index;
        break;
      }
    }
    if (failed_attempt_index < 0 &&
        failed_substitution.attempt_count < n8_attempt_capacity)
    {
      failed_attempt_index = failed_substitution.attempt_count++;
      failed_substitution.attempt_status[
          static_cast<std::size_t>(failed_attempt_index)] =
          sc25::TrajectoryStatus::HdgSc25InjectedLocalFactorFailure;
      failed_substitution.attempt_target_steps[
          static_cast<std::size_t>(failed_attempt_index)] =
          failed_substitution.step_numbers[0];
    }
    const int failed_target = failed_attempt_index >= 0 ?
        failed_substitution.attempt_target_steps[
            static_cast<std::size_t>(failed_attempt_index)] : -1;
    const int failed_local_step =
        failed_target - static_cast<int>(checkpoint_offset) - 1;
    if (failed_attempt_index >= 0 && failed_local_step >= 0 &&
        failed_local_step < failed_substitution.step_count)
    {
      failed_substitution.step_attempt_indices[
          static_cast<std::size_t>(failed_local_step)] =
          failed_attempt_index;
      TrajectoryRequire(
          !ValidAcceptedAttemptStepBijection(failed_substitution,
                                             checkpoint_offset),
          "trajectory failed-attempt substitution negative", evidence);
    }
    else
    {
      TrajectoryRequire(false,
                        "trajectory failed-attempt negative construction",
                        evidence);
    }

    if (attempt_step_mapping.step_count > 1)
    {
      AttemptStepMapping duplicate_attempt = attempt_step_mapping;
      duplicate_attempt.step_attempt_indices[1] =
          duplicate_attempt.step_attempt_indices[0];
      TrajectoryRequire(
          !ValidAcceptedAttemptStepBijection(duplicate_attempt,
                                             checkpoint_offset),
          "trajectory duplicate in-range attempt-index negative", evidence);
    }
  }
  TrajectoryRequire(
      run.continuation_start_coordinates.step == checkpoint_offset &&
          run.continuation_start_coordinates.generation ==
              checkpoint_offset &&
          run.continuation_start_coordinates.cursor == checkpoint_offset &&
          HashesEqual(IndependentHashes(run.continuation_start_state),
                      run.continuation_start_hashes),
      "trajectory continuation start state/coordinates/hashes", evidence);
  if (checkpoint_offset > 0)
  {
    TrajectoryRequire(run.restored_loaded_state_match &&
                          run.restored_loaded_coordinates_match &&
                          run.restored_loaded_hashes_match,
                      "trajectory immediate restored-load equality",
                      evidence);
  }
  TrajectoryRequire(run.coordinates.step == sc25::persistent_required_steps &&
                        run.coordinates.generation ==
                            sc25::persistent_required_steps &&
                        run.coordinates.cursor ==
                            sc25::persistent_required_steps,
                    "trajectory terminal coordinates", evidence);
  TrajectoryRequire(
      run.coordinates.dt == sc25::persistent_dt &&
          std::abs(run.coordinates.time -
                   sc25::persistent_required_steps * sc25::persistent_dt) <=
              8.0 * std::numeric_limits<double>::epsilon(),
      "trajectory terminal time", evidence);
  TrajectoryRequire(run.trace_restart_dimension ==
                        n8_trace_restart_dimension &&
                        run.trace_maximum_iterations ==
                            n8_trace_maximum_iterations &&
                        run.trace_relative_tolerance == 1.0e-12 &&
                        run.trace_absolute_tolerance == 1.0e-14,
                    "trajectory GMRES configuration", evidence);
  TrajectoryRequire(run.ilu_type == 0 && run.ilu_level_of_fill == 0 &&
                        run.ilu_local_reordering == 1,
                    "trajectory ILU(0)/RCM configuration", evidence);
  const sc25::AuxiliaryInitializationEvidence& initialization =
      run.initialization;
  TrajectoryRequire(
      initialization.quadrature_order ==
              sc25::coupled_fixture_quadrature_order &&
          initialization.trace_restart_dimension ==
              n8_trace_restart_dimension &&
          initialization.trace_maximum_iterations ==
              n8_trace_maximum_iterations &&
          initialization.trace_relative_tolerance == 1.0e-12 &&
          initialization.trace_absolute_tolerance == 1.0e-14 &&
          initialization.ilu_type == 0 &&
          initialization.ilu_level_of_fill == 0 &&
          initialization.ilu_local_reordering == 1,
      "trajectory auxiliary initialization configuration", evidence);
  TrajectoryRequire(
      initialization.trace_iterations >= 0 &&
          initialization.trace_iterations <=
              n8_trace_maximum_iterations &&
          std::isfinite(initialization.trace_true_residual_norm) &&
          std::isfinite(initialization.trace_right_hand_side_norm),
      "trajectory reported auxiliary trace-solve metadata", evidence);
  TrajectoryRequire(
      std::isfinite(initialization.residual.r_z_norm) &&
          std::isfinite(initialization.residual.r_sigma_norm) &&
          std::isfinite(initialization.residual.g_sigma_norm) &&
          initialization.residual.r_z_scale >= 1.0 &&
          initialization.residual.r_sigma_scale >= 1.0 &&
          initialization.residual.g_sigma_scale >= 1.0 &&
          std::isfinite(initialization.residual.normalized_maximum) &&
          initialization.residual.normalized_maximum <= 1.0e-10,
      "trajectory term-scaled auxiliary initialization residual", evidence);
  const sc25::TrajectoryReuseEvidence& reuse = run.reuse;
  const sc25::TrajectoryReuseSnapshot& baseline_reuse = reuse.baseline;
  TrajectoryRequire(baseline_reuse.topology_handle != 0 &&
                        baseline_reuse.foundation_handle != 0 &&
                        baseline_reuse.carrier_map_handle != 0 &&
                        baseline_reuse.contract_handle != 0 &&
                        baseline_reuse.assembly_state_handle != 0 &&
                        baseline_reuse.assembly_handle != 0 &&
                        baseline_reuse.trial_assembly_handle != 0 &&
                        baseline_reuse.factor_handle != 0 &&
                        baseline_reuse.workspace_handle != 0 &&
                        baseline_reuse.quadrature_workspace_handle != 0 &&
                        baseline_reuse.local_pattern_handle != 0 &&
                        baseline_reuse.true_matrix_handle != 0 &&
                        baseline_reuse.true_rhs_handle != 0 &&
                        baseline_reuse.true_solution_handle != 0 &&
                        baseline_reuse.ilu_handle != 0 &&
                        baseline_reuse.gmres_handle != 0 &&
                        baseline_reuse.true_residual_handle != 0 &&
                        baseline_reuse.element_to_true_csr_handle != 0,
                    "trajectory persistent handles", evidence);
  TrajectoryRequire(baseline_reuse.topology_generation == 1 &&
                        baseline_reuse.storage_generation == 1 &&
                        baseline_reuse.factor_generation == 1 &&
                        baseline_reuse.workspace_generation == 1 &&
                        baseline_reuse.local_pattern_generation == 1 &&
                        baseline_reuse.solver_generation == 1 &&
                        baseline_reuse.w8_owned_boundary_object_count > 0 &&
                        baseline_reuse.w8_owned_boundary_buffer_count > 0 &&
                        reuse.true_matrix_nonzeros > 0,
                    "trajectory reuse generations", evidence);
  TrajectoryRequire(ReuseSnapshotsEqual(baseline_reuse, reuse.current),
                    "trajectory stable owned storage", evidence);
  sc25::TrajectoryReuseSnapshot handle_drift = baseline_reuse;
  handle_drift.workspace_handle ^= std::uintptr_t{1};
  TrajectoryRequire(!ReuseSnapshotsEqual(handle_drift, baseline_reuse),
                    "trajectory copied handle-drift negative control",
                    evidence);
  sc25::TrajectoryReuseSnapshot capacity_hash_drift = baseline_reuse;
  ++capacity_hash_drift.w8_owned_boundary_buffer_count;
  capacity_hash_drift.owned_boundary_identity_hash.bytes[0] ^= 0x01U;
  TrajectoryRequire(
      !ReuseSnapshotsEqual(capacity_hash_drift, baseline_reuse),
      "trajectory copied capacity/hash-drift negative control", evidence);
  sc25::TrajectoryReuseSnapshot generation_drift = baseline_reuse;
  ++generation_drift.workspace_generation;
  TrajectoryRequire(!ReuseSnapshotsEqual(generation_drift, baseline_reuse),
                    "trajectory copied generation-drift negative control",
                    evidence);
  TrajectoryRequire(
      reuse.detected_w8_owned_boundary_identity_changes == 0 &&
          reuse.w8_device_transfer_calls == 0 &&
          !reuse.opaque_mfem_hypre_observed,
      "trajectory W8 boundary-census drift counters", evidence);

  N8OracleContext n8;
  if (!BuildN8OracleContext(foundation, n8, evidence)) { return false; }
  sc25::AuxiliaryResidualMetrics auxiliary_q12{}, auxiliary_q14{};
  const bool auxiliary_q12_valid = EvaluateIndependentAuxiliaryResidualImpl(
      foundation, run.continuation_start_state, n8_q12, auxiliary_q12);
  const bool auxiliary_q14_valid = EvaluateIndependentAuxiliaryResidualImpl(
      foundation, run.continuation_start_state, n8_q14, auxiliary_q14);
  const std::array<double, 3> auxiliary_normalized_q12{{
      auxiliary_q12.r_z_norm / auxiliary_q12.r_z_scale,
      auxiliary_q12.r_sigma_norm / auxiliary_q12.r_sigma_scale,
      auxiliary_q12.g_sigma_norm / auxiliary_q12.g_sigma_scale}};
  const std::array<double, 3> auxiliary_normalized_q14{{
      auxiliary_q14.r_z_norm / auxiliary_q14.r_z_scale,
      auxiliary_q14.r_sigma_norm / auxiliary_q14.r_sigma_scale,
      auxiliary_q14.g_sigma_norm / auxiliary_q14.g_sigma_scale}};
  bool auxiliary_sensitivity = true;
  for (std::size_t term = 0; term < auxiliary_normalized_q12.size(); ++term)
  {
    auxiliary_sensitivity = auxiliary_sensitivity &&
        std::abs(auxiliary_normalized_q14[term] -
                 auxiliary_normalized_q12[term]) <= 1.0e-11;
  }
  TrajectoryRequire(
      auxiliary_q12_valid && auxiliary_q14_valid &&
          auxiliary_q12.normalized_maximum <= 1.0e-10 &&
          auxiliary_q14.normalized_maximum <= 1.0e-10 &&
          auxiliary_sensitivity,
      "trajectory independent q12/q14 term-scaled auxiliary residual",
      evidence);
  if (checkpoint_offset == 0)
  {
    TrajectoryRequire(
        std::abs(initialization.residual.normalized_maximum -
                 auxiliary_q12.normalized_maximum) <= 2.0e-11,
        "trajectory reported/independent initialization residual",
        evidence);
  }
  if (checkpoint_offset == 0)
  {
    double q12_volume_moment = 0.0;
    double q12_trace_moment = 0.0;
    double q14_volume_moment = 0.0;
    double q14_trace_moment = 0.0;
    const bool q12_projection = VerifyIndependentInitialL2Projection(
        foundation, run.continuation_start_state, n8_q12,
        q12_volume_moment, q12_trace_moment);
    const bool q14_projection = VerifyIndependentInitialL2Projection(
        foundation, run.continuation_start_state, n8_q14,
        q14_volume_moment, q14_trace_moment);
    TrajectoryRequire(
        q12_projection && q14_projection &&
            q12_volume_moment <= 2.0e-12 &&
            q12_trace_moment <= 2.0e-12 &&
            q14_volume_moment <= 2.0e-12 &&
            q14_trace_moment <= 2.0e-12 &&
            std::abs(q14_volume_moment - q12_volume_moment) <= 1.0e-11 &&
            std::abs(q14_trace_moment - q12_trace_moment) <= 1.0e-11,
        "trajectory independent q12/q14 initial L2 projections", evidence);
  }

  std::array<double, sc25::state_components> preceding_integrals{};
  std::array<double, sc25::state_components> preceding_integrals_q14{};
  double start_density = 0.0;
  double start_pressure = 0.0;
  double start_margin = 0.0;
  double start_density_q14 = 0.0;
  double start_pressure_q14 = 0.0;
  double start_margin_q14 = 0.0;
  const bool have_preceding = IndependentVolumeChecks(
      foundation, run.continuation_start_state, preceding_integrals,
      start_density, start_pressure, start_margin, n8_q12) &&
      IndependentVolumeChecks(
          foundation, run.continuation_start_state,
          preceding_integrals_q14, start_density_q14,
          start_pressure_q14, start_margin_q14, n8_q14);
  TrajectoryRequire(have_preceding,
                    "trajectory independent continuation-start integrals",
                    evidence);
  for (int component = 0; component < sc25::state_components; ++component)
  {
    TrajectoryRequire(
        std::abs(preceding_integrals_q14[component] -
                 preceding_integrals[component]) /
                std::max(1.0, std::abs(preceding_integrals[component])) <=
            1.0e-11,
        "trajectory continuation-start q12/q14 integral sensitivity " +
            std::to_string(component), evidence);
  }
  for (int index = 0; index < run.step_evidence_count; ++index)
  {
    const sc25::TrajectoryStepEvidence& step =
        run.steps[static_cast<std::size_t>(index)];
    const int expected_step =
        static_cast<int>(checkpoint_offset) + index + 1;
    const std::string label =
        "trajectory step " + std::to_string(expected_step);
    TrajectoryRequire(step.accepted && step.step == expected_step &&
                          step.generation ==
                              static_cast<std::uint64_t>(expected_step) &&
                          step.dt == n8_dt &&
                          std::abs(step.time -
                                   expected_step * n8_dt) <=
                              8.0 * std::numeric_limits<double>::epsilon(),
                      label + " coordinates", evidence);
    TrajectoryRequire(FiniteState(step.accepted_state),
                      label + " finite state", evidence);
    std::vector<double> admissibility_residual, admissibility_residual_q14;
    const bool accepted_q12 = N8ResidualOnly(
        foundation, n8, step.accepted_state, step.accepted_state, n8_q12,
        admissibility_residual);
    const bool accepted_q14 = N8ResidualOnly(
        foundation, n8, step.accepted_state, step.accepted_state, n8_q14,
        admissibility_residual_q14);
    TrajectoryRequire(
        accepted_q12 && accepted_q14 &&
            C1Relative(admissibility_residual_q14,
                       admissibility_residual) <= 1.0e-11,
        label + " exact volume/side/trace quadrature admissibility", evidence);
    evidence.independently_assembled_residuals += 2;
    TrajectoryRequire(
        HashesEqual(IndependentHashes(step.accepted_state),
                    step.accepted_hashes),
        label + " independently reconstructed hashes", evidence);
    std::array<double, sc25::state_components> integrals{};
    std::array<double, sc25::state_components> integrals_q14{};
    double density = 0.0;
    double pressure = 0.0;
    double margin = 0.0;
    double density_q14 = 0.0;
    double pressure_q14 = 0.0;
    double margin_q14 = 0.0;
    const bool volume_valid = IndependentVolumeChecks(
        foundation, step.accepted_state, integrals, density, pressure, margin,
        n8_q12);
    const bool volume_q14_valid = IndependentVolumeChecks(
        foundation, step.accepted_state, integrals_q14, density_q14,
        pressure_q14, margin_q14, n8_q14);
    const bool trace_valid = volume_valid && IndependentTraceNodeChecks(
        step.accepted_state, density, pressure, margin);
    TrajectoryRequire(volume_valid && volume_q14_valid && trace_valid,
                      label + " independent admissibility", evidence);
    if (volume_valid && trace_valid)
    {
      evidence.minimum_density = std::min(evidence.minimum_density, density);
      evidence.minimum_pressure = std::min(evidence.minimum_pressure, pressure);
      evidence.minimum_speed_margin =
          std::min(evidence.minimum_speed_margin, margin);
    }
    double q12_conservation_maximum = 0.0;
    double q14_conservation_maximum = 0.0;
    for (int component = 0; component < sc25::state_components; ++component)
    {
      const double discrepancy =
          std::abs(integrals[component] - step.accepted_integrals[component]);
      evidence.maximum_integral_discrepancy =
          std::max(evidence.maximum_integral_discrepancy, discrepancy);
      TrajectoryRequire(discrepancy <= 2.0e-12,
                        label + " independent integral " +
                            std::to_string(component), evidence);
      const double quadrature_sensitivity =
          std::abs(integrals_q14[component] - integrals[component]) /
          std::max(1.0, std::abs(integrals[component]));
      TrajectoryRequire(
          quadrature_sensitivity <= 1.0e-11,
          label + " q12/q14 integral sensitivity " +
              std::to_string(component), evidence);
      const double conservation_error =
          std::abs(integrals[component] - preceding_integrals[component]);
      const double conservation_bound =
          2.0e-10 * std::max(1.0, std::abs(preceding_integrals[component]));
      TrajectoryRequire(
          conservation_error <= conservation_bound,
          label + " independently reconstructed conservation " +
              std::to_string(component), evidence);
      q12_conservation_maximum = std::max(
          q12_conservation_maximum,
          conservation_error /
              std::max(1.0, std::abs(preceding_integrals[component])));
      q14_conservation_maximum = std::max(
          q14_conservation_maximum,
          std::abs(integrals_q14[component] -
                   preceding_integrals_q14[component]) /
              std::max(1.0, std::abs(preceding_integrals_q14[component])));
      TrajectoryRequire(
          std::abs(step.committed_integrals[component] -
                   preceding_integrals[component]) <= 2.0e-12,
          label + " committed integral observation " +
              std::to_string(component), evidence);
    }
    const bool attempt_index_valid = step.attempt_index >= 0 &&
        step.attempt_index < run.attempt_count &&
        step.attempt_index < n8_attempt_capacity;
    const bool accepted_attempt_mapping = attempt_index_valid &&
        AcceptedAttemptStatus(run.attempts[
            static_cast<std::size_t>(step.attempt_index)].status) &&
        run.attempts[static_cast<std::size_t>(step.attempt_index)]
                .target_step == step.step;
    TrajectoryRequire(accepted_attempt_mapping,
                      label + " unique accepted-attempt mapping", evidence);
    if (accepted_attempt_mapping)
    {
      const double reported_q12 = run.attempts[
          static_cast<std::size_t>(step.attempt_index)]
                                      .maximum_conservation_error;
      evidence.maximum_conservation_maximum_discrepancy = std::max(
          evidence.maximum_conservation_maximum_discrepancy,
          std::max(std::abs(q12_conservation_maximum - reported_q12),
                   std::abs(q14_conservation_maximum - reported_q12)));
      TrajectoryRequire(
          ConservationMaximumMatches(q12_conservation_maximum,
                                     reported_q12),
          label + " independent q12/reported-q12 conservation maximum",
          evidence);
      TrajectoryRequire(
          ConservationMaximumMatches(q14_conservation_maximum,
                                     reported_q12),
          label + " independent q14/reported-q12 conservation maximum",
          evidence);
      const double away_from_q14 = reported_q12 >= q14_conservation_maximum ?
          reported_q12 + n8_conservation_negative_offset :
          reported_q12 - n8_conservation_negative_offset;
      TrajectoryRequire(
          !ConservationMaximumMatches(q14_conservation_maximum,
                                      away_from_q14),
          label + " reported conservation maximum sensitivity", evidence);
      ++evidence.checked_conservation_maxima;
    }
    preceding_integrals = integrals;
    preceding_integrals_q14 = integrals_q14;
    TrajectoryRequire(ReuseSnapshotsEqual(step.reuse, baseline_reuse),
                      label + " stable owned reuse snapshot", evidence);
    ++evidence.checked_steps;
  }
  TrajectoryRequire(
      evidence.checked_conservation_maxima == expected_continuation_steps,
      "trajectory conservation-maximum comparison census", evidence);

  TrajectoryRequire(run.attempt_count >= expected_continuation_steps &&
                        run.attempt_count <= expected_continuation_steps + 1,
                    "trajectory bounded attempt count", evidence);
  for (int index = 0; index < run.attempt_count; ++index)
  {
    const sc25::TrajectoryAttemptEvidence& attempt =
        run.attempts[static_cast<std::size_t>(index)];
    const std::string label = "trajectory attempt " + std::to_string(index);
    TrajectoryRequire(
        attempt.target_step > static_cast<int>(checkpoint_offset) &&
            attempt.target_step <= n8_steps && attempt.dt == n8_dt,
                      label + " identity", evidence);
    TrajectoryRequire(attempt.newton_iterations >= 0 &&
                          attempt.newton_iterations <= n8_newton_limit &&
                          attempt.attempted_newton_iterations >=
                              attempt.newton_iterations &&
                          attempt.attempted_newton_iterations <=
                              n8_newton_limit,
                      label + " Newton bound", evidence);
    TrajectoryRequire(
        ReuseSnapshotsEqual(attempt.entry_reuse, baseline_reuse) &&
            ReuseSnapshotsEqual(attempt.exit_reuse, baseline_reuse),
        label + " stable owned buffers across attempt", evidence);
    const TrajectoryState* committed = nullptr;
    const int local_target_index =
        attempt.target_step - static_cast<int>(checkpoint_offset) - 1;
    if (local_target_index > 0 &&
        local_target_index - 1 < run.step_evidence_count)
    {
      committed = &run.steps[static_cast<std::size_t>(
          local_target_index - 1)].accepted_state;
    }
    else if (local_target_index == 0)
    {
      committed = &run.continuation_start_state;
    }
    TrajectoryRequire(committed != nullptr, label + " committed snapshot",
                      evidence);
    const bool attempt_accepted = AcceptedAttemptStatus(attempt.status);
    if (attempt_accepted && local_target_index >= 0 &&
        local_target_index < run.step_evidence_count)
    {
      const sc25::TrajectoryStateHashes& accepted_hashes =
          run.steps[static_cast<std::size_t>(local_target_index)]
              .accepted_hashes;
      TrajectoryRequire(
          HashesEqual(attempt.candidate_hashes, accepted_hashes) &&
              HashesEqual(attempt.published_hashes, accepted_hashes),
          label + " accepted candidate/publication hashes", evidence);
    }
    else if (!attempt_accepted)
    {
      TrajectoryRequire(
          HashesEqual(attempt.entry_hashes, attempt.candidate_hashes) &&
              HashesEqual(attempt.entry_hashes,
                          attempt.published_hashes),
          label + " failed-attempt candidate/rollback hashes", evidence);
    }
    if (attempt.injection ==
        sc25::TrajectoryFailureInjection::newton_iteration_limit)
    {
      TrajectoryRequire(
          attempt.status ==
                  sc25::TrajectoryStatus::
                      HdgSc25InjectedNewtonIterationLimit &&
              attempt.attempted_newton_iterations == n8_newton_limit &&
              attempt.newton_iterations == 0,
          label + " exact injected Newton-limit counter/status", evidence);
    }
    if (committed)
    {
      std::vector<double> entry_residual;
      if (N8ResidualOnly(
              foundation, n8, *committed, *committed,
              sc25::coupled_fixture_quadrature_order, entry_residual))
      {
        const double oracle_norm = N8Norm(entry_residual);
        const double discrepancy = std::abs(
            oracle_norm - attempt.entry_residual_norm) /
            std::max(1.0, oracle_norm);
        evidence.maximum_residual_norm_discrepancy = std::max(
            evidence.maximum_residual_norm_discrepancy, discrepancy);
        TrajectoryRequire(discrepancy <= 2.0e-11,
                          label + " independent attempt-entry residual",
                          evidence);
      }
      else
      {
        TrajectoryRequire(false, label + " attempt-entry assembly", evidence);
      }
      ++evidence.independently_assembled_residuals;
    }
    for (int iteration = 0; iteration < attempt.newton_iterations;
         ++iteration)
    {
      const sc25::TrajectoryNewtonEvidence& newton =
          attempt.newton[static_cast<std::size_t>(iteration)];
      TrajectoryRequire(newton.iteration == iteration &&
                            std::isfinite(newton.entry_residual_norm) &&
                            newton.entry_residual_norm >= 0.0,
                        label + " Newton record", evidence);
      if (iteration == 0)
      {
        TrajectoryRequire(
            HashesEqual(IndependentHashes(newton.entry),
                        attempt.entry_hashes),
            label + " independently reconstructed entry hashes", evidence);
      }
      TrajectoryRequire(
          newton.trace_iterations >= 0 &&
              newton.trace_iterations <=
                  n8_trace_maximum_iterations &&
              std::isfinite(newton.trace_true_residual_norm) &&
              std::isfinite(newton.trace_right_hand_side_norm),
          label + " reported trace solve metadata", evidence);
      if (committed)
      {
        N8Assembly analytic, analytic_q14;
        std::vector<double> entry_residual, repeat_residual;
        const bool analytic_ok = N8AssembleAnalytic(
            foundation, n8, *committed, newton.entry, n8_q12, analytic);
        const bool analytic_q14_ok = N8AssembleAnalytic(
            foundation, n8, *committed, newton.entry, n8_q14,
            analytic_q14);
        const bool residual_ok = N8ResidualOnly(
            foundation, n8, *committed, newton.entry,
            sc25::coupled_fixture_quadrature_order, entry_residual);
        const bool repeat_ok = N8ResidualOnly(
            foundation, n8, *committed, newton.entry,
            sc25::coupled_fixture_repeat_quadrature_order, repeat_residual);
        TrajectoryRequire(
            analytic_ok && analytic_q14_ok && residual_ok && repeat_ok,
                          label + " independent Newton assembly", evidence);
        evidence.independently_assembled_residuals += 2;
        if (analytic_ok && analytic_q14_ok && residual_ok && repeat_ok)
        {
          const double entry_norm = N8Norm(entry_residual);
          const double norm_discrepancy =
              std::abs(entry_norm - newton.entry_residual_norm) /
              std::max(1.0, entry_norm);
          evidence.maximum_residual_norm_discrepancy = std::max(
              evidence.maximum_residual_norm_discrepancy, norm_discrepancy);
          TrajectoryRequire(C1Relative(analytic.residual, entry_residual) <=
                                2.0e-11,
                            label + " analytic/residual-only agreement",
                            evidence);
          TrajectoryRequire(C1Relative(repeat_residual, entry_residual) <=
                                1.0e-11,
                            label + " quadrature-repeat residual", evidence);
          TrajectoryRequire(N8AssemblyRelative(analytic_q14, analytic) <=
                                1.0e-11,
                            label + " q12/q14 Jacobian/block sensitivity",
                            evidence);
          TrajectoryRequire(norm_discrepancy <= 2.0e-11,
                            label + " independent Newton-entry residual",
                            evidence);
          VerifyN8Direction(n8, analytic, newton.direction, evidence, label);
          VerifyN8Direction(n8, analytic_q14, newton.direction, evidence,
                            label + " q14");
        }

        int independent_first_armijo = -1;
        double independent_accepted_norm =
            std::numeric_limits<double>::infinity();
        for (int line = 0; line < sc25::persistent_line_search_trials; ++line)
        {
          const double theta = std::ldexp(1.0, -line);
          const TrajectoryState trial = ApplyIndependentDirection(
              newton.entry, newton.direction, theta);
          std::vector<double> trial_residual, trial_residual_q14;
          const bool independently_admissible = N8ResidualOnly(
              foundation, n8, *committed, trial, n8_q12, trial_residual);
          const bool independently_admissible_q14 = N8ResidualOnly(
              foundation, n8, *committed, trial, n8_q14,
              trial_residual_q14);
          evidence.independently_assembled_residuals += 2;
          TrajectoryRequire(
              independently_admissible == independently_admissible_q14 &&
                  (!independently_admissible ||
                   C1Relative(trial_residual_q14, trial_residual) <=
                       1.0e-11),
              label + " trial q12/q14 residual/card sensitivity", evidence);
          const double trial_norm = independently_admissible ?
              N8Norm(trial_residual) :
              std::numeric_limits<double>::infinity();
          const double entry_norm = analytic_ok ? N8Norm(analytic.residual) :
              std::numeric_limits<double>::infinity();
          const bool armijo = independently_admissible &&
              std::isfinite(trial_norm) &&
              trial_norm <= (1.0 - 1.0e-4 * theta) * entry_norm;
          if (independent_first_armijo < 0 && armijo)
          {
            independent_first_armijo = line;
            independent_accepted_norm = trial_norm;
          }
          const bool injection_override = attempt.injection ==
              sc25::TrajectoryFailureInjection::admissibility;
          if (injection_override)
          {
            TrajectoryRequire(
                newton.trial_admissible[static_cast<std::size_t>(line)] == 0 &&
                    newton.trial_armijo[static_cast<std::size_t>(line)] == 0,
                label + " all-trials-inadmissible hook", evidence);
          }
          else if (line <= newton.accepted_line_search_index)
          {
            TrajectoryRequire(
                independently_admissible ==
                    (newton.trial_admissible[static_cast<std::size_t>(line)] !=
                     0),
                label + " independent trial admissibility", evidence);
            const double reported = newton.trial_residual_norms[
                static_cast<std::size_t>(line)];
            const double discrepancy = std::abs(reported - trial_norm) /
                std::max(1.0, trial_norm);
            evidence.maximum_residual_norm_discrepancy = std::max(
                evidence.maximum_residual_norm_discrepancy, discrepancy);
            TrajectoryRequire(discrepancy <= 2.0e-11,
                              label + " independent trial residual", evidence);
            TrajectoryRequire(
                armijo ==
                    (newton.trial_armijo[static_cast<std::size_t>(line)] != 0),
                label + " independent Armijo decision", evidence);
          }
          ++evidence.checked_line_search_trials;
        }
        if (attempt.injection ==
            sc25::TrajectoryFailureInjection::admissibility)
        {
          TrajectoryRequire(newton.accepted_line_search_index == -1,
                            label + " injected line-search rejection",
                            evidence);
        }
        else
        {
          TrajectoryRequire(independent_first_armijo >= 0 &&
                                newton.accepted_line_search_index ==
                                    independent_first_armijo,
                            label + " independently first Armijo trial",
                            evidence);
          const double discrepancy =
              std::abs(newton.accepted_residual_norm -
                       independent_accepted_norm) /
              std::max(1.0, independent_accepted_norm);
          TrajectoryRequire(discrepancy <= 2.0e-11,
                            label + " independently accepted residual",
                            evidence);
        }
      }
      ++evidence.checked_newton_iterations;
    }
    if (attempt_accepted && committed && local_target_index >= 0 &&
        local_target_index < run.step_evidence_count)
    {
      std::vector<double> independent_attempt_entry;
      std::vector<double> final_residual, final_residual_q14;
      const TrajectoryState& accepted_state = run.steps[
          static_cast<std::size_t>(local_target_index)].accepted_state;
      const bool entry_ok = N8ResidualOnly(
          foundation, n8, *committed, *committed, n8_q12,
          independent_attempt_entry);
      const bool final_q12 = N8ResidualOnly(
          foundation, n8, *committed, accepted_state, n8_q12,
          final_residual);
      const bool final_q14 = N8ResidualOnly(
          foundation, n8, *committed, accepted_state, n8_q14,
          final_residual_q14);
      if (entry_ok && final_q12 && final_q14)
      {
        const double entry_norm = N8Norm(independent_attempt_entry);
        const double final_norm = N8Norm(final_residual);
        const double discrepancy =
            std::abs(final_norm - attempt.final_residual_norm) /
            std::max(1.0, final_norm);
        evidence.maximum_residual_norm_discrepancy = std::max(
            evidence.maximum_residual_norm_discrepancy, discrepancy);
        TrajectoryRequire(discrepancy <= 2.0e-11,
                          label + " independent accepted residual", evidence);
        TrajectoryRequire(
            C1Relative(final_residual_q14, final_residual) <= 1.0e-11,
            label + " accepted q12/q14 residual sensitivity", evidence);
        TrajectoryRequire(
            final_norm <= 1.0e-12 + 1.0e-10 * entry_norm,
            label + " independent final nonlinear residual bound", evidence);
      }
      else
      {
        TrajectoryRequire(false, label + " accepted-state assembly", evidence);
      }
      evidence.independently_assembled_residuals += 3;
    }
    ++evidence.checked_attempts;
  }
  return evidence.failures.empty();
}

namespace
{

void Rank2Require(const bool condition, const std::string& message,
                  IndependentRank2TrajectoryEvidence& evidence)
{
  if (!condition) { evidence.failures.push_back(message); }
}

int ElementAtCenter(mfem::ParMesh& mesh, const double x,
                    const double y)
{
  const mfem::IntegrationPoint& center =
      mfem::Geometries.GetCenter(mfem::Geometry::SQUARE);
  mfem::Vector point(sc25::dimension);
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    mfem::ElementTransformation* transformation =
        mesh.GetElementTransformation(element);
    if (!transformation) { continue; }
    transformation->Transform(center, point);
    if (std::abs(point[0] - x) <= 1.0e-13 &&
        std::abs(point[1] - y) <= 1.0e-13)
    {
      return element;
    }
  }
  return -1;
}

struct FaceSample final
{
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::array<double, sc25::trace_components> values{};
};

std::int64_t PeriodicCoordinateKey(double value)
{
  value -= std::floor(value);
  if (std::abs(value - 1.0) <= 1.0e-12 ||
      std::abs(value) <= 1.0e-12)
  {
    value = 0.0;
  }
  return static_cast<std::int64_t>(std::llround(value * 1.0e12));
}

bool BuildFaceSamples(
    sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const mfem::Vector& true_values, std::vector<FaceSample>& samples)
{
  if (!foundation.SetTraceCarrierFromTrueDofs(true_values)) { return false; }
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  const sc25::FieldOwnershipView fields = foundation.fields();
  const mfem::ParFiniteElementSpace* space = spaces.trace_carrier.address;
  const mfem::ParGridFunction* field = fields.trace_carrier;
  if (!mesh || !space || !field) { return false; }
  samples.clear();
  samples.reserve(static_cast<std::size_t>(2 * mesh->GetNumFaces()));
  constexpr std::array<double, 2> points = {
      0.21132486540518711775, 0.78867513459481288225};
  for (int face = 0; face < mesh->GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* finite_element = space->GetFaceElement(face);
    mfem::ElementTransformation* transformation =
        mesh->GetFaceTransformation(face);
    mfem::Array<int> signed_vdofs;
    space->GetFaceVDofs(face, signed_vdofs);
    if (!finite_element || !transformation || finite_element->GetDof() != 2 ||
        signed_vdofs.Size() != 2 * sc25::trace_components)
    {
      return false;
    }
    for (const double coordinate : points)
    {
      mfem::IntegrationPoint ip;
      ip.Set(coordinate, 0.0, 0.0, 1.0);
      mfem::Vector shape(2), physical(sc25::dimension);
      finite_element->CalcShape(ip, shape);
      transformation->Transform(ip, physical);
      FaceSample sample;
      sample.x = PeriodicCoordinateKey(physical[0]);
      sample.y = PeriodicCoordinateKey(physical[1]);
      for (int component = 0; component < sc25::trace_components;
           ++component)
      {
        for (int scalar = 0; scalar < 2; ++scalar)
        {
          int decoded = -1;
          int sign = 0;
          if (!DecodeSignedDof(signed_vdofs[component * 2 + scalar],
                               decoded, sign) ||
              decoded < 0 || decoded >= field->Size())
          {
            return false;
          }
          sample.values[static_cast<std::size_t>(component)] +=
              shape[scalar] * sign * (*field)[decoded];
        }
      }
      samples.push_back(sample);
    }
  }
  return true;
}

const FaceSample* FindFaceSample(const std::vector<FaceSample>& samples,
                                 const FaceSample& target)
{
  const auto iterator = std::find_if(
      samples.begin(), samples.end(), [&target](const FaceSample& candidate)
      { return candidate.x == target.x && candidate.y == target.y; });
  return iterator == samples.end() ? nullptr : &*iterator;
}

double Rank2PeerStateRelativeError(
    sc25::PersistentSc25ScalarIgrCpuFoundation& rank2,
    sc25::PersistentSc25ScalarIgrCpuFoundation& rank1_peer,
    const sc25::PersistentRank2TrajectoryState& local_state,
    const TrajectoryState& peer_state,
    IndependentRank2TrajectoryEvidence& evidence,
    const std::string_view label)
{
  mfem::ParMesh* local_mesh = const_cast<mfem::ParMesh*>(rank2.mesh());
  mfem::ParMesh* peer_mesh = const_cast<mfem::ParMesh*>(rank1_peer.mesh());
  if (!local_mesh || !peer_mesh)
  {
    Rank2Require(false, std::string(label) + " mesh availability", evidence);
    return std::numeric_limits<double>::infinity();
  }
  const mfem::IntegrationPoint& center =
      mfem::Geometries.GetCenter(mfem::Geometry::SQUARE);
  mfem::Vector physical(sc25::dimension);
  double local_error = 0.0;
  for (int local_element = 0; local_element < local_mesh->GetNE();
       ++local_element)
  {
    mfem::ElementTransformation* transformation =
        local_mesh->GetElementTransformation(local_element);
    if (!transformation)
    {
      Rank2Require(false, std::string(label) + " element transformation",
                   evidence);
      continue;
    }
    transformation->Transform(center, physical);
    const int peer_element =
        ElementAtCenter(*peer_mesh, physical[0], physical[1]);
    if (peer_element < 0)
    {
      Rank2Require(false, std::string(label) + " geometric element match",
                   evidence);
      continue;
    }
    const std::size_t local_u = static_cast<std::size_t>(local_element) * 16;
    const std::size_t peer_u = static_cast<std::size_t>(peer_element) * 16;
    const std::size_t local_sigma =
        static_cast<std::size_t>(local_element) * 4;
    const std::size_t peer_sigma =
        static_cast<std::size_t>(peer_element) * 4;
    const std::size_t local_z = static_cast<std::size_t>(local_element) * 8;
    const std::size_t peer_z = static_cast<std::size_t>(peer_element) * 8;
    local_error = std::max(
        local_error,
        RelativeL2(local_state.u.data() + local_u,
                   peer_state.u.data() + peer_u, 16));
    local_error = std::max(
        local_error,
        RelativeL2(local_state.sigma.data() + local_sigma,
                   peer_state.sigma.data() + peer_sigma, 4));
    local_error = std::max(
        local_error,
        RelativeL2(local_state.z.data() + local_z,
                   peer_state.z.data() + peer_z, 8));
  }

  mfem::Vector local_true(
      static_cast<int>(local_state.uhat.size() +
                       local_state.sigmahat.size()));
  std::copy(local_state.uhat.begin(), local_state.uhat.end(),
            local_true.HostWrite());
  std::copy(local_state.sigmahat.begin(), local_state.sigmahat.end(),
            local_true.HostWrite() + local_state.uhat.size());
  mfem::Vector peer_true(sc25::persistent_true_trace_unknowns);
  std::copy(peer_state.uhat.begin(), peer_state.uhat.end(),
            peer_true.HostWrite());
  std::copy(peer_state.sigmahat.begin(), peer_state.sigmahat.end(),
            peer_true.HostWrite() + peer_state.uhat.size());
  std::vector<FaceSample> local_samples, peer_samples;
  if (!BuildFaceSamples(rank2, local_true, local_samples) ||
      !BuildFaceSamples(rank1_peer, peer_true, peer_samples))
  {
    Rank2Require(false, std::string(label) + " trace sampling", evidence);
    local_error = std::numeric_limits<double>::infinity();
  }
  else
  {
    for (const FaceSample& sample : local_samples)
    {
      const FaceSample* peer = FindFaceSample(peer_samples, sample);
      if (!peer)
      {
        Rank2Require(false, std::string(label) + " physical face match",
                     evidence);
        local_error = std::numeric_limits<double>::infinity();
        continue;
      }
      local_error = std::max(
          local_error,
          RelativeL2(sample.values.data(), peer->values.data(),
                     sc25::trace_components));
    }
  }
  double global_error = local_error;
  if (MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    Rank2Require(false, std::string(label) + " error reduction", evidence);
    return std::numeric_limits<double>::infinity();
  }
  return global_error;
}

double Rank2RelativeScalar(const double left, const double right) noexcept
{
  return std::abs(left - right) /
      std::max({1.0, std::abs(left), std::abs(right)});
}

double PeerStateNorm(const TrajectoryState& state,
                     const bool trace_only) noexcept
{
  long double square = 0.0L;
  const auto add = [&square](const auto& values)
  {
    for (const double value : values)
    {
      square += static_cast<long double>(value) * value;
    }
  };
  if (!trace_only)
  {
    add(state.u);
    add(state.sigma);
    add(state.z);
  }
  add(state.uhat);
  add(state.sigmahat);
  return std::sqrt(static_cast<double>(square));
}

bool IndependentlyCheckRank2Restriction(
    const sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    IndependentRank2TrajectoryEvidence& evidence)
{
  const mfem::ParMesh* mesh = foundation.mesh();
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  const mfem::ParFiniteElementSpace* carrier = spaces.trace_carrier.address;
  const mfem::DG_Interface_FECollection* collection =
      foundation.trace_collection();
  const auto& production = foundation.element_topology();
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (!mesh || !carrier || !collection ||
      production.size() != static_cast<std::size_t>(mesh->GetNE()) ||
      ranks != 2)
  {
    return false;
  }

  const int local_true_size = carrier->GetTrueVSize();
  std::array<int, 2> true_sizes{};
  if (MPI_Allgather(&local_true_size, 1, MPI_INT, true_sizes.data(), 1,
                    MPI_INT, MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  const std::array<dg::hdg::GlobalIndex, 3> true_offsets = {
      0, static_cast<dg::hdg::GlobalIndex>(true_sizes[0]),
      static_cast<dg::hdg::GlobalIndex>(true_sizes[0] + true_sizes[1])};
  if (true_offsets[2] != sc25::persistent_rank2_global_true_trace_unknowns)
  {
    return false;
  }

  std::vector<int> local_face_ids;
  local_face_ids.reserve(static_cast<std::size_t>(mesh->GetNumFaces()));
  std::vector<int> local_face_geometric_owner(
      static_cast<std::size_t>(mesh->GetNumFaces()), -1);
  int independently_owned_faces = 0;
  const mfem::IntegrationPoint& face_center =
      mfem::Geometries.GetCenter(mfem::Geometry::SEGMENT);
  mfem::Vector physical(sc25::dimension);
  mfem::ParMesh* mutable_mesh = const_cast<mfem::ParMesh*>(mesh);
  for (int face = 0; face < mesh->GetNumFaces(); ++face)
  {
    mfem::ElementTransformation* transformation =
        mutable_mesh->GetFaceTransformation(face);
    if (!transformation) { return false; }
    transformation->Transform(face_center, physical);
    double x = physical[0] - std::floor(physical[0]);
    double y = physical[1] - std::floor(physical[1]);
    if (std::abs(x - 1.0) <= 1.0e-12) { x = 0.0; }
    if (std::abs(y - 1.0) <= 1.0e-12) { y = 0.0; }
    const double scaled_x = 8.0 * x;
    const double scaled_y = 8.0 * y;
    const bool vertical =
        std::abs(scaled_x - std::round(scaled_x)) <= 1.0e-11;
    const bool horizontal =
        std::abs(scaled_y - std::round(scaled_y)) <= 1.0e-11;
    if (vertical == horizontal) { return false; }
    int face_id = -1;
    int geometric_owner = -1;
    if (vertical)
    {
      const int line = static_cast<int>(std::llround(scaled_x)) % 8;
      const int cell = static_cast<int>(std::floor(scaled_y));
      const int left_cell = (line + 7) % 8;
      const int right_cell = line;
      geometric_owner = std::min(left_cell < 4 ? 0 : 1,
                                 right_cell < 4 ? 0 : 1);
      face_id = line * 8 + cell;
    }
    else
    {
      const int line = static_cast<int>(std::llround(scaled_y)) % 8;
      const int cell = static_cast<int>(std::floor(scaled_x));
      geometric_owner = cell < 4 ? 0 : 1;
      face_id = 64 + line * 8 + cell;
    }
    if (face_id < 0 || face_id >= 128 || geometric_owner < 0)
    {
      return false;
    }
    local_face_ids.push_back(face_id);
    local_face_geometric_owner[static_cast<std::size_t>(face)] =
        geometric_owner;
    independently_owned_faces += geometric_owner == rank ? 1 : 0;
    mfem::Array<int> face_vdofs;
    carrier->GetFaceVDofs(face, face_vdofs);
    if (face_vdofs.Size() != 2 * sc25::trace_components)
    {
      return false;
    }
    for (int face_row = 0; face_row < face_vdofs.Size(); ++face_row)
    {
      int local_vdof = -1;
      int carrier_sign = 0;
      if (!DecodeSignedDof(face_vdofs[face_row], local_vdof,
                           carrier_sign) ||
          local_vdof < 0 || local_vdof >= carrier->GetVSize())
      {
        return false;
      }
      const dg::hdg::GlobalIndex global_true_dof =
          static_cast<dg::hdg::GlobalIndex>(
              carrier->GetGlobalTDofNumber(local_vdof));
      const int tdof_owner = global_true_dof >= true_offsets[0] &&
              global_true_dof < true_offsets[1] ? 0 :
          global_true_dof >= true_offsets[1] &&
              global_true_dof < true_offsets[2] ? 1 : -1;
      Rank2Require(
          tdof_owner == geometric_owner,
          "rank2 physical face " + std::to_string(face_id) +
              " carrier row " + std::to_string(face_row) +
              " global TDof owner binding geometric=" +
              std::to_string(geometric_owner) + " mfem=" +
              std::to_string(tdof_owner), evidence);
      ++evidence.checked_owner_bound_face_dofs;
    }
  }
  std::sort(local_face_ids.begin(), local_face_ids.end());
  if (local_face_ids.size() != 72 ||
      std::adjacent_find(local_face_ids.begin(), local_face_ids.end()) !=
          local_face_ids.end())
  {
    return false;
  }
  std::array<int, 144> gathered_face_ids{};
  if (MPI_Allgather(local_face_ids.data(), 72, MPI_INT,
                    gathered_face_ids.data(), 72, MPI_INT,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  std::array<int, 128> face_multiplicity{};
  for (const int face_id : gathered_face_ids)
  {
    if (face_id < 0 || face_id >= 128) { return false; }
    ++face_multiplicity[static_cast<std::size_t>(face_id)];
  }
  const int global_faces = static_cast<int>(std::count_if(
      face_multiplicity.begin(), face_multiplicity.end(),
      [](const int count) { return count != 0; }));
  const int shared_faces = static_cast<int>(std::count(
      face_multiplicity.begin(), face_multiplicity.end(), 2));
  Rank2Require(global_faces == 128 && shared_faces == 16 &&
                   std::all_of(face_multiplicity.begin(),
                               face_multiplicity.end(),
                               [](const int count)
                               { return count == 1 || count == 2; }),
               "rank2 independent periodic face census", evidence);
  evidence.independently_owned_faces = independently_owned_faces;
  evidence.independently_owned_true_dofs = independently_owned_faces * 10;
  evidence.independently_global_faces = global_faces;

  std::vector<double> global_x(
      sc25::persistent_rank2_global_true_trace_unknowns);
  std::vector<long double> local_pty(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0L);
  std::vector<long double> global_pty(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0L);
  std::vector<int> local_coverage(
      sc25::persistent_rank2_global_true_trace_unknowns, 0);
  std::vector<int> global_coverage(
      sc25::persistent_rank2_global_true_trace_unknowns, 0);
  for (std::size_t entry = 0; entry < global_x.size(); ++entry)
  {
    global_x[entry] = 0.125 + static_cast<double>(entry) / 2048.0;
  }
  long double local_px_dot_y = 0.0L;
  for (int element = 0; element < mesh->GetNE(); ++element)
  {
    mfem::Array<int> faces, orientations;
    mesh->GetElementEdges(element, faces, orientations);
    if (faces.Size() != sc25::faces_per_quadrilateral ||
        orientations.Size() != faces.Size())
    {
      return false;
    }
    const sc25::ElementTopology& actual =
        production[static_cast<std::size_t>(element)];
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      mfem::Array<int> face_vdofs;
      carrier->GetFaceVDofs(faces[local_face], face_vdofs);
      const int* permutation = collection->DofOrderForOrientation(
          mfem::Geometry::SEGMENT, orientations[local_face]);
      if (!permutation ||
          face_vdofs.Size() != 2 * sc25::trace_components)
      {
        return false;
      }
      for (int component = 0; component < sc25::trace_components;
           ++component)
        for (int oriented = 0; oriented < 2; ++oriented)
        {
          int canonical = -1;
          int orientation_sign = 0;
          int local_vdof = -1;
          int carrier_sign = 0;
          if (!DecodeSignedDof(permutation[oriented], canonical,
                               orientation_sign) ||
              !DecodeSignedDof(face_vdofs[component * 2 + canonical],
                               local_vdof, carrier_sign) ||
              local_vdof < 0 ||
              local_vdof >= carrier->GetVSize())
          {
            return false;
          }
          const int sign = orientation_sign * carrier_sign;
          const int row = component * 8 + local_face * 2 + oriented;
          const int signed_vdof = sign < 0 ? -1 - local_vdof : local_vdof;
          const dg::hdg::GlobalIndex global_true_dof =
              static_cast<dg::hdg::GlobalIndex>(
                  carrier->GetGlobalTDofNumber(local_vdof));
          if (global_true_dof < 0 || global_true_dof >= true_offsets[2])
          {
            return false;
          }
          const dg::hdg::RankIndex owner_rank =
              global_true_dof < true_offsets[1] ? 0 : 1;
          const int mfem_local_true_dof =
              carrier->GetLocalTDofNumber(local_vdof);
          const dg::hdg::LocalIndex local_true_dof = owner_rank == rank ?
              static_cast<dg::hdg::LocalIndex>(mfem_local_true_dof) :
              dg::hdg::invalid_local_index;
          if ((owner_rank == rank &&
               (mfem_local_true_dof < 0 ||
                global_true_dof != true_offsets[rank] +
                    mfem_local_true_dof)) ||
              (owner_rank != rank && mfem_local_true_dof >= 0))
          {
            return false;
          }
          const sc25::TraceRestrictionEntry& item =
              actual.restriction[static_cast<std::size_t>(row)];
          Rank2Require(
              item.element == element && item.local_row == row &&
                  item.local_face == local_face &&
                  item.component == component &&
                  item.oriented_face_dof == oriented &&
                  item.canonical_face_dof == canonical &&
                  item.signed_carrier_vdof == signed_vdof &&
                  item.decoded_carrier_vdof == local_vdof &&
                  item.sign == sign &&
                  item.global_true_dof == global_true_dof &&
                  item.local_true_dof == local_true_dof &&
                  item.owner_rank == owner_rank,
              "rank2 independent signed restriction row", evidence);
          const long double y =
              0.25L + static_cast<long double>(rank * 1280 +
                                               element * 40 + row) /
                  4096.0L;
          const long double px = static_cast<long double>(sign) *
              global_x[static_cast<std::size_t>(global_true_dof)];
          local_px_dot_y += px * y;
          local_pty[static_cast<std::size_t>(global_true_dof)] +=
              static_cast<long double>(sign) * y;
          ++local_coverage[static_cast<std::size_t>(global_true_dof)];
          ++evidence.checked_restriction_rows;
        }
    }
  }
  long double global_px_dot_y = 0.0L;
  if (MPI_Allreduce(&local_px_dot_y, &global_px_dot_y, 1, MPI_LONG_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(local_pty.data(), global_pty.data(),
                    static_cast<int>(global_pty.size()), MPI_LONG_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(local_coverage.data(), global_coverage.data(),
                    static_cast<int>(global_coverage.size()), MPI_INT,
                    MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  long double global_x_dot_pty = 0.0L;
  for (std::size_t entry = 0; entry < global_pty.size(); ++entry)
  {
    global_x_dot_pty +=
        static_cast<long double>(global_x[entry]) * global_pty[entry];
  }
  const long double adjoint_scale = std::max(
      1.0L, std::max(std::abs(global_px_dot_y),
                     std::abs(global_x_dot_pty)));
  evidence.p_pt_adjoint_relative_error = static_cast<double>(
      std::abs(global_px_dot_y - global_x_dot_pty) / adjoint_scale);
  evidence.independently_covered_true_dofs = static_cast<int>(std::count(
      global_coverage.begin(), global_coverage.end(), 2));
  Rank2Require(evidence.p_pt_adjoint_relative_error <= 5.0e-15,
               "rank2 independent P/Pt adjoint action", evidence);
  Rank2Require(evidence.independently_covered_true_dofs == 1280 &&
                   std::all_of(global_coverage.begin(), global_coverage.end(),
                               [](const int count) { return count == 2; }),
               "rank2 independent true-DOF restriction coverage", evidence);
  return true;
}

struct Rank2IndependentContext final
{
  std::vector<IndependentElementRestriction> restriction;
  std::array<int, 2> true_sizes{};
  std::array<int, 2> true_offsets{};
};

struct Rank2IndependentAssembly final
{
  std::vector<sc25::ElementLinearization> elements;
  std::vector<double> residual;
};

bool BuildRank2IndependentContext(
    const sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    Rank2IndependentContext& context)
{
  context = {};
  const mfem::ParMesh* mesh = foundation.mesh();
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  const mfem::ParFiniteElementSpace* carrier =
      spaces.trace_carrier.address;
  const mfem::DG_Interface_FECollection* collection =
      foundation.trace_collection();
  if (!mesh || !carrier || !collection || mesh->GetNE() != 32)
  {
    return false;
  }
  const int local_true = carrier->GetTrueVSize();
  if (MPI_Allgather(&local_true, 1, MPI_INT, context.true_sizes.data(), 1,
                    MPI_INT, MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  context.true_offsets = {{0, context.true_sizes[0]}};
  if (context.true_sizes[0] + context.true_sizes[1] !=
      sc25::persistent_rank2_global_true_trace_unknowns)
  {
    return false;
  }
  context.restriction.resize(static_cast<std::size_t>(mesh->GetNE()));
  for (int element = 0; element < mesh->GetNE(); ++element)
  {
    IndependentElementRestriction& output =
        context.restriction[static_cast<std::size_t>(element)];
    output = {};
    output.element = element;
    output.scalar_volume_dofs = c1_ns;
    output.scalar_face_dofs = c1_face_nodes;
    output.local_interior_unknowns = c1_ni;
    output.local_trace_unknowns = c1_nt;
    mfem::Array<int> faces, orientations;
    mesh->GetElementEdges(element, faces, orientations);
    if (faces.Size() != c1_faces_per_element ||
        orientations.Size() != faces.Size())
    {
      return false;
    }
    output.faces.assign(faces.begin(), faces.end());
    output.orientations.assign(orientations.begin(), orientations.end());
    output.entries.resize(c1_nt);
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      mfem::Array<int> face_vdofs;
      carrier->GetFaceVDofs(faces[local_face], face_vdofs);
      const int* permutation = collection->DofOrderForOrientation(
          mfem::Geometry::SEGMENT, orientations[local_face]);
      if (!permutation ||
          face_vdofs.Size() != c1_face_nodes * c1_trace_components)
      {
        return false;
      }
      for (int component = 0; component < c1_trace_components; ++component)
        for (int oriented = 0; oriented < c1_face_nodes; ++oriented)
        {
          int canonical = -1;
          int orientation_sign = 0;
          int carrier_vdof = -1;
          int carrier_sign = 0;
          if (!DecodeSignedDof(permutation[oriented], canonical,
                               orientation_sign) ||
              !DecodeSignedDof(
                  face_vdofs[component * c1_face_nodes + canonical],
                  carrier_vdof, carrier_sign))
          {
            return false;
          }
          const auto global_true =
              carrier->GetGlobalTDofNumber(carrier_vdof);
          const int owner_rank = global_true < context.true_sizes[0] ? 0 : 1;
          const int local_tdof = carrier->GetLocalTDofNumber(carrier_vdof);
          const int sign = orientation_sign * carrier_sign;
          const int row = component * c1_nf +
              local_face * c1_face_nodes + oriented;
          if (carrier_vdof < 0 || carrier_vdof >= carrier->GetVSize() ||
              global_true < 0 ||
              global_true >= sc25::persistent_rank2_global_true_trace_unknowns ||
              sign == 0)
          {
            return false;
          }
          output.entries[static_cast<std::size_t>(row)] = {
              element, row, local_face, component, oriented, canonical,
              sign < 0 ? -1 - carrier_vdof : carrier_vdof, carrier_vdof,
              sign, static_cast<dg::hdg::GlobalIndex>(global_true),
              local_tdof >= 0 ?
                  static_cast<dg::hdg::LocalIndex>(local_tdof) :
                  dg::hdg::invalid_local_index,
              static_cast<dg::hdg::RankIndex>(owner_rank)};
        }
    }
  }
  return true;
}

bool GatherRank2Trace(
    const sc25::PersistentRank2TrajectoryState& state,
    const Rank2IndependentContext& context,
    std::vector<double>& global_trace)
{
  std::vector<double> local(state.uhat.size() + state.sigmahat.size());
  std::copy(state.uhat.begin(), state.uhat.end(), local.begin());
  std::copy(state.sigmahat.begin(), state.sigmahat.end(),
            local.begin() + state.uhat.size());
  global_trace.assign(sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  const std::array<int, 2> displacements = {{0, context.true_sizes[0]}};
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank >= 0 && rank < 2 &&
      local.size() ==
          static_cast<std::size_t>(context.true_sizes[rank]) &&
      MPI_Allgatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                     global_trace.data(), context.true_sizes.data(),
                     displacements.data(), MPI_DOUBLE,
                     MPI_COMM_WORLD) == MPI_SUCCESS;
}

void PackRank2Interior(const sc25::PersistentRank2TrajectoryState& state,
                       const int element, std::vector<double>& local)
{
  local.assign(c1_ni, 0.0);
  const std::size_t u_base = static_cast<std::size_t>(element) * 16;
  const std::size_t sigma_base = static_cast<std::size_t>(element) * 4;
  const std::size_t z_base = static_cast<std::size_t>(element) * 8;
  std::copy_n(state.u.begin() + u_base, 16, local.begin());
  std::copy_n(state.sigma.begin() + sigma_base, 4, local.begin() + 16);
  std::copy_n(state.z.begin() + z_base, 8, local.begin() + 20);
}

bool AssembleIndependentRank2(
    const sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const Rank2IndependentContext& context,
    const sc25::PersistentRank2TrajectoryState& old_state,
    const sc25::PersistentRank2TrajectoryState& candidate,
    Rank2IndependentAssembly& output)
{
  output = {};
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  const mfem::DG_Interface_FECollection* collection =
      foundation.trace_collection();
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (!mesh || !spaces.sigma.address || !spaces.trace_carrier.address ||
      !collection || context.restriction.size() != 32 || rank < 0)
  {
    return false;
  }
  std::vector<double> old_trace_global, candidate_trace_global;
  if (!GatherRank2Trace(old_state, context, old_trace_global) ||
      !GatherRank2Trace(candidate, context, candidate_trace_global))
  {
    return false;
  }
  output.elements.resize(32);
  output.residual.assign(
      sc25::persistent_rank2_local_interior_unknowns +
          context.true_sizes[static_cast<std::size_t>(rank)],
      0.0);
  std::vector<double> local_trace_residual(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  std::vector<double> global_trace_residual(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  for (int element = 0; element < 32; ++element)
  {
    const IndependentElementRestriction& map =
        context.restriction[static_cast<std::size_t>(element)];
    std::vector<double> old, current_values;
    PackRank2Interior(old_state, element, old);
    PackRank2Interior(candidate, element, current_values);
    std::vector<double> old_trace(c1_nt), trace_values(c1_nt);
    for (int row = 0; row < c1_nt; ++row)
    {
      const IndependentRestrictionEntry& entry =
          map.entries[static_cast<std::size_t>(row)];
      old_trace[static_cast<std::size_t>(row)] = entry.sign *
          old_trace_global[static_cast<std::size_t>(entry.global_true_dof)];
      trace_values[static_cast<std::size_t>(row)] = entry.sign *
          candidate_trace_global[
              static_cast<std::size_t>(entry.global_true_dof)];
    }
    std::vector<C1Dual> current(c1_ni), trace(c1_nt);
    for (int column = 0; column < c1_ni; ++column)
    {
      current[static_cast<std::size_t>(column)] = C1Dual::Variable(
          current_values[static_cast<std::size_t>(column)], column);
    }
    for (int column = 0; column < c1_nt; ++column)
    {
      trace[static_cast<std::size_t>(column)] = C1Dual::Variable(
          trace_values[static_cast<std::size_t>(column)], c1_ni + column);
    }
    std::vector<C1Dual> local_residual, trace_residual;
    if (!C1ElementResidual(
            *mesh, *spaces.sigma.address, *spaces.trace_carrier.address,
            *collection, map, old, old_trace, current, trace, n8_q12,
            local_residual, trace_residual))
    {
      return false;
    }
    sc25::ElementLinearization& block =
        output.elements[static_cast<std::size_t>(element)];
    block.element = element;
    block.local_interior_unknowns = c1_ni;
    block.local_trace_unknowns = c1_nt;
    block.a.assign(static_cast<std::size_t>(c1_ni) * c1_ni, 0.0);
    block.b.assign(static_cast<std::size_t>(c1_ni) * c1_nt, 0.0);
    block.c.assign(static_cast<std::size_t>(c1_nt) * c1_ni, 0.0);
    block.d.assign(static_cast<std::size_t>(c1_nt) * c1_nt, 0.0);
    block.local_residual.resize(c1_ni);
    block.trace_residual.resize(c1_nt);
    for (int row = 0; row < c1_ni; ++row)
    {
      block.local_residual[static_cast<std::size_t>(row)] =
          local_residual[static_cast<std::size_t>(row)].value;
      output.residual[static_cast<std::size_t>(element) * c1_ni + row] =
          block.local_residual[static_cast<std::size_t>(row)];
      for (int column = 0; column < c1_ni; ++column)
      {
        block.a[static_cast<std::size_t>(row) * c1_ni + column] =
            local_residual[static_cast<std::size_t>(row)]
                .derivative[static_cast<std::size_t>(column)];
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        block.b[static_cast<std::size_t>(row) * c1_nt + column] =
            local_residual[static_cast<std::size_t>(row)]
                .derivative[static_cast<std::size_t>(c1_ni + column)];
      }
    }
    for (int row = 0; row < c1_nt; ++row)
    {
      block.trace_residual[static_cast<std::size_t>(row)] =
          trace_residual[static_cast<std::size_t>(row)].value;
      const IndependentRestrictionEntry& entry =
          map.entries[static_cast<std::size_t>(row)];
      local_trace_residual[
          static_cast<std::size_t>(entry.global_true_dof)] +=
          entry.sign * block.trace_residual[static_cast<std::size_t>(row)];
      for (int column = 0; column < c1_ni; ++column)
      {
        block.c[static_cast<std::size_t>(row) * c1_ni + column] =
            trace_residual[static_cast<std::size_t>(row)]
                .derivative[static_cast<std::size_t>(column)];
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        block.d[static_cast<std::size_t>(row) * c1_nt + column] =
            trace_residual[static_cast<std::size_t>(row)]
                .derivative[static_cast<std::size_t>(c1_ni + column)];
      }
    }
  }
  if (MPI_Allreduce(local_trace_residual.data(), global_trace_residual.data(),
                    static_cast<int>(global_trace_residual.size()),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  const int offset = context.true_offsets[static_cast<std::size_t>(rank)];
  std::copy_n(global_trace_residual.begin() + offset,
              context.true_sizes[static_cast<std::size_t>(rank)],
              output.residual.begin() +
                  sc25::persistent_rank2_local_interior_unknowns);
  return true;
}

void MakeIndependentAuxiliaryAssembly(Rank2IndependentAssembly& assembly)
{
  for (sc25::ElementLinearization& block : assembly.elements)
  {
    const auto a = block.a;
    const auto b = block.b;
    const auto c = block.c;
    const auto d = block.d;
    const auto local = block.local_residual;
    const auto trace = block.trace_residual;
    std::fill(block.a.begin(), block.a.end(), 0.0);
    std::fill(block.b.begin(), block.b.end(), 0.0);
    std::fill(block.c.begin(), block.c.end(), 0.0);
    std::fill(block.d.begin(), block.d.end(), 0.0);
    std::fill(block.local_residual.begin(), block.local_residual.end(), 0.0);
    std::fill(block.trace_residual.begin(), block.trace_residual.end(), 0.0);
    for (int row = 0; row < 16; ++row)
    {
      block.a[static_cast<std::size_t>(row) * c1_ni + row] = 1.0;
    }
    for (int row = 0; row < 32; ++row)
    {
      block.d[static_cast<std::size_t>(row) * c1_nt + row] = 1.0;
    }
    for (int row = 16; row < c1_ni; ++row)
    {
      block.local_residual[static_cast<std::size_t>(row)] =
          local[static_cast<std::size_t>(row)];
      for (int column = 16; column < c1_ni; ++column)
      {
        block.a[static_cast<std::size_t>(row) * c1_ni + column] =
            a[static_cast<std::size_t>(row) * c1_ni + column];
      }
      for (int column = 32; column < c1_nt; ++column)
      {
        block.b[static_cast<std::size_t>(row) * c1_nt + column] =
            b[static_cast<std::size_t>(row) * c1_nt + column];
      }
    }
    for (int row = 32; row < c1_nt; ++row)
    {
      block.trace_residual[static_cast<std::size_t>(row)] =
          trace[static_cast<std::size_t>(row)];
      for (int column = 16; column < c1_ni; ++column)
      {
        block.c[static_cast<std::size_t>(row) * c1_ni + column] =
            c[static_cast<std::size_t>(row) * c1_ni + column];
      }
      for (int column = 32; column < c1_nt; ++column)
      {
        block.d[static_cast<std::size_t>(row) * c1_nt + column] =
            d[static_cast<std::size_t>(row) * c1_nt + column];
      }
    }
  }
}

bool ExpectedRank2OperatorVectors(
    const sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const Rank2IndependentContext& context,
    const Rank2IndependentAssembly& assembly,
    std::vector<double>& residual, std::vector<double>& action_input,
    std::vector<double>& action_result,
    std::vector<double>& condensed_rhs)
{
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  if (rank < 0 || rank >= 2 || !mesh || assembly.elements.size() != 32 ||
      context.restriction.size() != 32)
  {
    return false;
  }
  const int owned = context.true_sizes[static_cast<std::size_t>(rank)];
  const int offset = context.true_offsets[static_cast<std::size_t>(rank)];
  const std::size_t full_size =
      sc25::persistent_rank2_local_interior_unknowns +
      static_cast<std::size_t>(owned);
  residual.assign(full_size, 0.0);
  action_input.assign(full_size, 0.0);
  action_result.assign(full_size, 0.0);
  condensed_rhs.assign(static_cast<std::size_t>(owned), 0.0);
  std::vector<double> local_trace_residual(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  std::vector<double> local_trace_action(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  std::vector<double> local_trace_rhs(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  std::vector<double> global_trace_residual(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  std::vector<double> global_trace_action(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  std::vector<double> global_trace_rhs(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  const mfem::IntegrationPoint& center =
      mfem::Geometries.GetCenter(mfem::Geometry::SQUARE);
  mfem::Vector physical(sc25::dimension);
  for (int element = 0; element < 32; ++element)
  {
    const sc25::ElementLinearization& block =
        assembly.elements[static_cast<std::size_t>(element)];
    const IndependentElementRestriction& map =
        context.restriction[static_cast<std::size_t>(element)];
    mfem::ElementTransformation* transformation =
        mesh->GetElementTransformation(element);
    if (!transformation) { return false; }
    transformation->Transform(center, physical);
    const int ex = std::max(0, std::min(
        sc25::persistent_elements_per_direction - 1,
        static_cast<int>(std::floor(
            physical[0] * sc25::persistent_elements_per_direction))));
    const int ey = std::max(0, std::min(
        sc25::persistent_elements_per_direction - 1,
        static_cast<int>(std::floor(
            physical[1] * sc25::persistent_elements_per_direction))));
    const int global_element =
        ey * sc25::persistent_elements_per_direction + ex;
    const std::size_t base = static_cast<std::size_t>(element) * c1_ni;
    std::array<double, c1_nt> trace_action{};
    for (int row = 0; row < c1_nt; ++row)
    {
      const IndependentRestrictionEntry& entry =
          map.entries[static_cast<std::size_t>(row)];
      const double global_value = -0.015625 +
          static_cast<double>(entry.global_true_dof + 1) / 131072.0;
      trace_action[static_cast<std::size_t>(row)] =
          entry.sign * global_value;
    }
    for (int row = 0; row < c1_ni; ++row)
    {
      residual[base + row] =
          block.local_residual[static_cast<std::size_t>(row)];
      action_input[base + row] = 0.03125 +
          static_cast<double>(global_element * c1_ni + row + 1) / 65536.0;
      double value = 0.0;
      for (int column = 0; column < c1_ni; ++column)
      {
        const double action = 0.03125 +
            static_cast<double>(global_element * c1_ni + column + 1) /
                65536.0;
        value += block.a[static_cast<std::size_t>(row) * c1_ni + column] *
            action;
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        value += block.b[static_cast<std::size_t>(row) * c1_nt + column] *
            trace_action[static_cast<std::size_t>(column)];
      }
      action_result[base + row] = value;
    }
    C1Lu factor;
    std::vector<double> inverse_residual;
    if (!factor.Factor(block.a, c1_ni) ||
        !factor.Solve(block.local_residual, inverse_residual))
    {
      return false;
    }
    for (int row = 0; row < c1_nt; ++row)
    {
      const IndependentRestrictionEntry& entry =
          map.entries[static_cast<std::size_t>(row)];
      local_trace_residual[
          static_cast<std::size_t>(entry.global_true_dof)] +=
          entry.sign * block.trace_residual[static_cast<std::size_t>(row)];
      double action = 0.0;
      for (int column = 0; column < c1_ni; ++column)
      {
        const double input = 0.03125 +
            static_cast<double>(global_element * c1_ni + column + 1) /
                65536.0;
        action += block.c[static_cast<std::size_t>(row) * c1_ni + column] *
            input;
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        action += block.d[static_cast<std::size_t>(row) * c1_nt + column] *
            trace_action[static_cast<std::size_t>(column)];
      }
      local_trace_action[
          static_cast<std::size_t>(entry.global_true_dof)] +=
          entry.sign * action;
      double rhs = -block.trace_residual[static_cast<std::size_t>(row)];
      for (int inner = 0; inner < c1_ni; ++inner)
      {
        rhs += block.c[static_cast<std::size_t>(row) * c1_ni + inner] *
            inverse_residual[static_cast<std::size_t>(inner)];
      }
      local_trace_rhs[static_cast<std::size_t>(entry.global_true_dof)] +=
          entry.sign * rhs;
    }
  }
  if (MPI_Allreduce(local_trace_residual.data(), global_trace_residual.data(),
                    static_cast<int>(global_trace_residual.size()),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(local_trace_action.data(), global_trace_action.data(),
                    static_cast<int>(global_trace_action.size()),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(local_trace_rhs.data(), global_trace_rhs.data(),
                    static_cast<int>(global_trace_rhs.size()),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  for (int row = 0; row < owned; ++row)
  {
    const std::size_t target =
        sc25::persistent_rank2_local_interior_unknowns +
        static_cast<std::size_t>(row);
    const int global = offset + row;
    residual[target] = global_trace_residual[static_cast<std::size_t>(global)];
    action_input[target] = -0.015625 +
        static_cast<double>(global + 1) / 131072.0;
    action_result[target] =
        global_trace_action[static_cast<std::size_t>(global)];
    condensed_rhs[static_cast<std::size_t>(row)] =
        global_trace_rhs[static_cast<std::size_t>(global)];
  }
  return true;
}

bool VerifyRank2SolveAndRecovery(
    const Rank2IndependentContext& context,
    const Rank2IndependentAssembly& assembly,
    const sc25::Rank2TrajectoryNewtonEvidence& item,
    double& trace_solve_relative_error,
    double& recovery_relative_error)
{
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank < 0 || rank >= 2 || assembly.elements.size() != 32 ||
      item.true_trace_solution.size() !=
          static_cast<std::size_t>(context.true_sizes[rank]))
  {
    return false;
  }
  std::vector<double> global_solution(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  const std::array<int, 2> displacements = {{0, context.true_sizes[0]}};
  if (MPI_Allgatherv(
          item.true_trace_solution.data(),
          static_cast<int>(item.true_trace_solution.size()), MPI_DOUBLE,
          global_solution.data(), context.true_sizes.data(),
          displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  std::vector<double> expected_interior(
      sc25::persistent_rank2_local_interior_unknowns, 0.0);
  std::vector<double> reported_interior(
      sc25::persistent_rank2_local_interior_unknowns, 0.0);
  std::vector<double> local_condensed_residual(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  std::vector<double> global_condensed_residual(
      sc25::persistent_rank2_global_true_trace_unknowns, 0.0);
  long double local_rhs_square = 0.0L;
  for (const double value : item.condensed_right_hand_side)
  {
    local_rhs_square += static_cast<long double>(value) * value;
  }
  for (int element = 0; element < 32; ++element)
  {
    const sc25::ElementLinearization& block =
        assembly.elements[static_cast<std::size_t>(element)];
    const IndependentElementRestriction& map =
        context.restriction[static_cast<std::size_t>(element)];
    C1Lu factor;
    std::vector<double> inverse_residual;
    if (!factor.Factor(block.a, c1_ni) ||
        !factor.Solve(block.local_residual, inverse_residual))
    {
      return false;
    }
    std::vector<double> inverse_b(
        static_cast<std::size_t>(c1_ni) * c1_nt, 0.0);
    for (int column = 0; column < c1_nt; ++column)
    {
      std::vector<double> rhs(c1_ni), solution;
      for (int row = 0; row < c1_ni; ++row)
      {
        rhs[static_cast<std::size_t>(row)] =
            block.b[static_cast<std::size_t>(row) * c1_nt + column];
      }
      if (!factor.Solve(rhs, solution)) { return false; }
      for (int row = 0; row < c1_ni; ++row)
      {
        inverse_b[static_cast<std::size_t>(row) * c1_nt + column] =
            solution[static_cast<std::size_t>(row)];
      }
    }
    std::array<double, c1_nt> local_trace_solution{};
    for (int row = 0; row < c1_nt; ++row)
    {
      const IndependentRestrictionEntry& entry =
          map.entries[static_cast<std::size_t>(row)];
      local_trace_solution[static_cast<std::size_t>(row)] = entry.sign *
          global_solution[static_cast<std::size_t>(entry.global_true_dof)];
    }
    std::vector<double> recovery_rhs = block.local_residual;
    for (int row = 0; row < c1_ni; ++row)
      for (int column = 0; column < c1_nt; ++column)
      {
        recovery_rhs[static_cast<std::size_t>(row)] +=
            block.b[static_cast<std::size_t>(row) * c1_nt + column] *
            local_trace_solution[static_cast<std::size_t>(column)];
      }
    std::vector<double> recovered;
    if (!factor.Solve(recovery_rhs, recovered)) { return false; }
    const std::size_t base = static_cast<std::size_t>(element) * c1_ni;
    for (int row = 0; row < c1_ni; ++row)
    {
      expected_interior[base + row] =
          -recovered[static_cast<std::size_t>(row)];
    }
    std::copy_n(item.recovered_increment.u.begin() + element * 16, 16,
                reported_interior.begin() + base);
    std::copy_n(item.recovered_increment.sigma.begin() + element * 4, 4,
                reported_interior.begin() + base + 16);
    std::copy_n(item.recovered_increment.z.begin() + element * 8, 8,
                reported_interior.begin() + base + 20);
    for (int row = 0; row < c1_nt; ++row)
    {
      double value = block.trace_residual[static_cast<std::size_t>(row)];
      for (int inner = 0; inner < c1_ni; ++inner)
      {
        value -= block.c[static_cast<std::size_t>(row) * c1_ni + inner] *
            inverse_residual[static_cast<std::size_t>(inner)];
      }
      for (int column = 0; column < c1_nt; ++column)
      {
        double schur =
            block.d[static_cast<std::size_t>(row) * c1_nt + column];
        for (int inner = 0; inner < c1_ni; ++inner)
        {
          schur -= block.c[static_cast<std::size_t>(row) * c1_ni + inner] *
              inverse_b[static_cast<std::size_t>(inner) * c1_nt + column];
        }
        value += schur *
            local_trace_solution[static_cast<std::size_t>(column)];
      }
      const IndependentRestrictionEntry& entry =
          map.entries[static_cast<std::size_t>(row)];
      local_condensed_residual[
          static_cast<std::size_t>(entry.global_true_dof)] +=
          entry.sign * value;
    }
  }
  if (MPI_Allreduce(local_condensed_residual.data(),
                    global_condensed_residual.data(),
                    static_cast<int>(global_condensed_residual.size()),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  long double local_solution_residual_square = 0.0L;
  const int offset = context.true_offsets[static_cast<std::size_t>(rank)];
  for (int row = 0; row < context.true_sizes[rank]; ++row)
  {
    const double value =
        global_condensed_residual[static_cast<std::size_t>(offset + row)];
    local_solution_residual_square +=
        static_cast<long double>(value) * value;
  }
  std::array<long double, 2> local_squares = {
      local_solution_residual_square, local_rhs_square};
  std::array<long double, 2> global_squares{};
  if (MPI_Allreduce(local_squares.data(), global_squares.data(), 2,
                    MPI_LONG_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  trace_solve_relative_error = std::sqrt(
      static_cast<double>(global_squares[0])) /
      std::max(1.0, std::sqrt(static_cast<double>(global_squares[1])));
  recovery_relative_error = C1Relative(expected_interior,
                                       reported_interior);
  std::vector<double> reported_trace(
      item.recovered_increment.uhat.size() +
      item.recovered_increment.sigmahat.size());
  std::copy(item.recovered_increment.uhat.begin(),
            item.recovered_increment.uhat.end(), reported_trace.begin());
  std::copy(item.recovered_increment.sigmahat.begin(),
            item.recovered_increment.sigmahat.end(),
            reported_trace.begin() + item.recovered_increment.uhat.size());
  recovery_relative_error = std::max(
      recovery_relative_error,
      C1Relative(item.true_trace_solution, reported_trace));
  return true;
}

class Rank2ManifestBuilder final
{
public:
  bool Text(const std::string_view value) noexcept
  {
    return U64(value.size()) && hash_.Update(value.data(), value.size());
  }

  bool U64(const std::uint64_t value) noexcept
  {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
      bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
    return hash_.Update(bytes.data(), bytes.size());
  }

  bool Hash(const dg::hdg::runtime::StateHash& value) noexcept
  {
    return hash_.Update(value.bytes.data(), value.bytes.size());
  }

  dg::hdg::runtime::StateHash Finish() noexcept
  {
    dg::hdg::runtime::Sha256Digest digest{};
    return hash_.Finalize(digest) ? dg::hdg::runtime::StateHash{digest} :
                                    dg::hdg::runtime::StateHash{};
  }

private:
  dg::hdg::runtime::Sha256 hash_;
};

sc25::Rank2TrajectoryStateHashes IndependentRank2Hashes(
    const sc25::PersistentRank2TrajectoryState& state) noexcept
{
  return {HashOneArray(state.u),
          HashOneArray(state.sigma),
          HashOneArray(state.z),
          HashOneArray(state.uhat),
          HashOneArray(state.sigmahat),
          HashArraySequence(state.u, state.sigma, state.z),
          HashArraySequence(state.uhat, state.sigmahat),
          HashArraySequence(state.u, state.sigma, state.z, state.uhat,
                            state.sigmahat)};
}

} // namespace

bool EvaluateIndependentRank2StateAdmissibilityOracle(
    sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
    const sc25::PersistentRank2TrajectoryState& state,
    const int quadrature_order)
{
  const sc25::SpaceOwnershipSnapshot spaces = foundation.spaces();
  const sc25::FieldOwnershipView fields = foundation.fields();
  mfem::ParMesh* mesh = const_cast<mfem::ParMesh*>(foundation.mesh());
  if (!mesh || !spaces.u.address || !spaces.trace_carrier.address ||
      !fields.trace_carrier || mesh->GetNE() != 32)
  {
    return false;
  }
  mfem::Vector true_state(
      static_cast<int>(state.uhat.size() + state.sigmahat.size()));
  std::copy(state.uhat.begin(), state.uhat.end(), true_state.HostWrite());
  std::copy(state.sigmahat.begin(), state.sigmahat.end(),
            true_state.HostWrite() + state.uhat.size());
  if (!foundation.SetTraceCarrierFromTrueDofs(true_state)) { return false; }
  bool local_admissible = true;
  const auto sample = [](const std::array<double, sc25::state_components>& u)
  {
    double density = 0.0;
    double pressure = 0.0;
    double margin = 0.0;
    return PrimitiveAdmissible(u, density, pressure, margin);
  };
  try
  {
    for (int element = 0; local_admissible && element < mesh->GetNE();
         ++element)
    {
      const mfem::FiniteElement* finite_element =
          spaces.u.address->GetFE(element);
      mfem::ElementTransformation* transformation =
          mesh->GetElementTransformation(element);
      if (!finite_element || !transformation ||
          finite_element->GetDof() != sc25::persistent_scalar_volume_nodes)
      {
        local_admissible = false;
        break;
      }
      mfem::Vector shape(sc25::persistent_scalar_volume_nodes);
      const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), quadrature_order);
      const std::size_t u_base = static_cast<std::size_t>(element) * 16;
      for (int point = 0; point < volume_rule.GetNPoints(); ++point)
      {
        finite_element->CalcShape(volume_rule.IntPoint(point), shape);
        std::array<double, sc25::state_components> conservative{};
        for (int component = 0; component < sc25::state_components;
             ++component)
          for (int scalar = 0;
               scalar < sc25::persistent_scalar_volume_nodes; ++scalar)
          {
            conservative[static_cast<std::size_t>(component)] +=
                shape[scalar] * state.u[
                    u_base + component *
                        sc25::persistent_scalar_volume_nodes + scalar];
          }
        if (!sample(conservative))
        {
          local_admissible = false;
          break;
        }
      }
      mfem::Array<int> faces, orientations;
      mesh->GetElementEdges(element, faces, orientations);
      for (int local_face = 0;
           local_admissible && local_face < faces.Size(); ++local_face)
      {
        const int face = faces[local_face];
        mfem::FaceElementTransformations* face_transformations =
            mesh->GetFaceElementTransformations(face);
        const mfem::FiniteElement* trace_element =
            spaces.trace_carrier.address->GetFaceElement(face);
        mfem::Array<int> signed_vdofs;
        spaces.trace_carrier.address->GetFaceVDofs(face, signed_vdofs);
        if (!face_transformations || !trace_element ||
            trace_element->GetDof() != 2 ||
            signed_vdofs.Size() != 2 * sc25::trace_components)
        {
          local_admissible = false;
          break;
        }
        const bool element_one = face_transformations->Elem1No == element;
        if (!element_one && face_transformations->Elem2No != element)
        {
          local_admissible = false;
          break;
        }
        mfem::Vector interior_shape(
            sc25::persistent_scalar_volume_nodes), trace_shape(2);
        const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
            trace_element->GetGeomType(), quadrature_order);
        for (int point = 0; point < face_rule.GetNPoints(); ++point)
        {
          const mfem::IntegrationPoint& ip = face_rule.IntPoint(point);
          face_transformations->SetAllIntPoints(&ip);
          const mfem::IntegrationPoint& element_ip = element_one ?
              face_transformations->GetElement1IntPoint() :
              face_transformations->GetElement2IntPoint();
          finite_element->CalcShape(element_ip, interior_shape);
          trace_element->CalcShape(ip, trace_shape);
          std::array<double, sc25::state_components> conservative{},
              trace_state{};
          for (int component = 0; component < sc25::state_components;
               ++component)
          {
            for (int scalar = 0;
                 scalar < sc25::persistent_scalar_volume_nodes; ++scalar)
            {
              conservative[static_cast<std::size_t>(component)] +=
                  interior_shape[scalar] * state.u[
                      u_base + component *
                          sc25::persistent_scalar_volume_nodes + scalar];
            }
            for (int scalar = 0; scalar < 2; ++scalar)
            {
              int dof = -1;
              int sign = 0;
              if (!DecodeSignedDof(
                      signed_vdofs[component * 2 + scalar], dof, sign) ||
                  dof < 0 || dof >= fields.trace_carrier->Size())
              {
                local_admissible = false;
                break;
              }
              trace_state[static_cast<std::size_t>(component)] +=
                  trace_shape[scalar] * sign *
                  (*fields.trace_carrier)[dof];
            }
          }
          if (!local_admissible || !sample(conservative) ||
              !sample(trace_state))
          {
            local_admissible = false;
            break;
          }
        }
      }
    }
    // Evaluate the independent carrier field on every physical face, not
    // merely the element-side restriction paths above.
    for (int face = 0; local_admissible && face < mesh->GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* trace_element =
          spaces.trace_carrier.address->GetFaceElement(face);
      mfem::Array<int> signed_vdofs;
      spaces.trace_carrier.address->GetFaceVDofs(face, signed_vdofs);
      if (!trace_element || trace_element->GetDof() != 2 ||
          signed_vdofs.Size() != 2 * sc25::trace_components)
      {
        local_admissible = false;
        break;
      }
      mfem::Vector shape(2);
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          trace_element->GetGeomType(), quadrature_order);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
        trace_element->CalcShape(rule.IntPoint(point), shape);
        std::array<double, sc25::state_components> trace_state{};
        for (int component = 0; component < sc25::state_components;
             ++component)
          for (int scalar = 0; scalar < 2; ++scalar)
          {
            int dof = -1;
            int sign = 0;
            if (!DecodeSignedDof(signed_vdofs[component * 2 + scalar],
                                 dof, sign) ||
                dof < 0 || dof >= fields.trace_carrier->Size())
            {
              local_admissible = false;
              break;
            }
            trace_state[static_cast<std::size_t>(component)] +=
                shape[scalar] * sign * (*fields.trace_carrier)[dof];
          }
        if (!local_admissible || !sample(trace_state))
        {
          local_admissible = false;
          break;
        }
      }
    }
  }
  catch (...)
  {
    local_admissible = false;
  }
  const int local = local_admissible ? 1 : 0;
  int global = 0;
  return MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && global == 1;
}

bool RunIndependentRank2TrajectoryOracle(
    sc25::PersistentSc25ScalarIgrCpuFoundation& rank2,
    sc25::PersistentSc25ScalarIgrCpuFoundation& rank1_peer,
    const sc25::PersistentRank2TrajectoryCheckpoint& checkpoint10,
    IndependentRank2TrajectoryEvidence& evidence)
{
  evidence = {};
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  mfem::ParMesh* local_mesh = const_cast<mfem::ParMesh*>(rank2.mesh());
  mfem::ParMesh* peer_mesh = const_cast<mfem::ParMesh*>(rank1_peer.mesh());
  const sc25::SpaceOwnershipSnapshot local_spaces = rank2.spaces();
  const sc25::PersistentRank2TrajectoryEvidence& local_run =
      rank2.rank2_trajectory_evidence();
  const sc25::PersistentTrajectoryEvidence& peer_run =
      rank1_peer.trajectory_evidence();
  Rank2Require(ranks == 2 && local_mesh && peer_mesh,
               "rank2 independent communicator", evidence);
  if (!local_mesh || !peer_mesh) { return false; }

  IndependentTrajectoryEvidence peer_oracle;
  if (!RunIndependentTrajectoryOracle(rank1_peer, peer_oracle))
  {
    for (const std::string& failure : peer_oracle.failures)
    {
      Rank2Require(false, "rank1 scientific oracle: " + failure, evidence);
    }
  }
  evidence.independent_rank1_residual_assemblies =
      peer_oracle.independently_assembled_residuals;

  evidence.checked_local_elements = local_mesh->GetNE();
  evidence.checked_local_faces = local_mesh->GetNumFaces();
  Rank2Require(local_mesh->GetNE() == 32 && local_mesh->GetNumFaces() == 72,
               "rank2 independent local mesh census", evidence);
  if (!IndependentlyCheckRank2Restriction(rank2, evidence))
  {
    Rank2Require(false, "rank2 independent restriction construction",
                 evidence);
  }
  Rank2IndependentContext rank2_context;
  Rank2Require(BuildRank2IndependentContext(rank2, rank2_context),
               "rank2 separately authored distributed restriction",
               evidence);
  const auto check_operator_vectors =
      [&](const sc25::Rank2TrajectoryNewtonEvidence& item,
          const Rank2IndependentAssembly& assembly,
          const std::string& label)
  {
    std::vector<double> expected_residual, expected_action_input,
        expected_action_result, expected_rhs;
    const bool expected = ExpectedRank2OperatorVectors(
        rank2, rank2_context, assembly, expected_residual,
        expected_action_input, expected_action_result, expected_rhs);
    Rank2Require(expected, label + " expected operator vectors", evidence);
    if (!expected) { return; }
    const std::array<double, 4> discrepancies = {
        C1Relative(item.nonlinear_residual, expected_residual),
        C1Relative(item.jacobian_action_input, expected_action_input),
        C1Relative(item.jacobian_action_result, expected_action_result),
        C1Relative(item.condensed_right_hand_side, expected_rhs)};
    for (const double discrepancy : discrepancies)
    {
      evidence.maximum_vector_relative_error = std::max(
          evidence.maximum_vector_relative_error, discrepancy);
    }
    Rank2Require(
        std::all_of(discrepancies.begin(), discrepancies.end(),
                    [](const double value) { return value <= 5.0e-11; }),
        label + " independent residual/action/condensed-RHS vectors",
        evidence);
    double trace_solve_error = std::numeric_limits<double>::infinity();
    double recovery_error = std::numeric_limits<double>::infinity();
    const bool solve_recovery = item.completed &&
        VerifyRank2SolveAndRecovery(rank2_context, assembly, item,
                                    trace_solve_error, recovery_error);
    Rank2Require(solve_recovery,
                 label + " independent trace solve/recovery", evidence);
    if (solve_recovery)
    {
      evidence.maximum_vector_relative_error = std::max(
          {evidence.maximum_vector_relative_error, trace_solve_error,
           recovery_error});
      Rank2Require(trace_solve_error <= 5.0e-11 &&
                       recovery_error <= 5.0e-11,
                   label + " true-trace solution/recovered vector",
                   evidence);
    }
    evidence.checked_vector_records += 6;
  };
  const int independently_ghost_true_dofs =
      evidence.checked_local_faces * 10 - evidence.independently_owned_true_dofs;
  Rank2Require(
      local_spaces.trace_carrier.local_true_size ==
              evidence.independently_owned_true_dofs &&
          local_spaces.trace_carrier.global_true_size ==
              sc25::persistent_rank2_global_true_trace_unknowns,
      "rank2 MFEM true-space census versus independent geometry", evidence);
  Rank2Require(
      local_run.census.rank == rank && local_run.census.ranks == ranks &&
          local_run.census.local_elements == evidence.checked_local_elements &&
          local_run.census.local_interior_unknowns == 896 &&
          local_run.census.local_faces == evidence.checked_local_faces &&
          local_run.census.owned_unique_faces ==
              evidence.independently_owned_faces &&
          local_run.census.combined_owned_trace_dofs ==
              evidence.independently_owned_true_dofs &&
          local_run.census.combined_ghost_trace_dofs ==
              independently_ghost_true_dofs &&
          local_run.census.global_unique_faces ==
              evidence.independently_global_faces &&
          local_run.census.global_combined_true_dofs ==
              evidence.independently_covered_true_dofs,
      "rank2 published census versus independent reconstruction", evidence);

  const sc25::AuxiliaryInitializationEvidence& local_initialization =
      local_run.initialization;
  const sc25::AuxiliaryInitializationEvidence& peer_initialization =
      peer_run.initialization;
  Rank2Require(
      local_initialization.quadrature_order ==
              peer_initialization.quadrature_order &&
          local_initialization.trace_restart_dimension ==
              peer_initialization.trace_restart_dimension &&
          local_initialization.trace_maximum_iterations ==
              peer_initialization.trace_maximum_iterations &&
          local_initialization.ilu_type == peer_initialization.ilu_type &&
          local_initialization.ilu_level_of_fill ==
              peer_initialization.ilu_level_of_fill &&
          local_initialization.ilu_local_reordering ==
              peer_initialization.ilu_local_reordering &&
          local_initialization.trace_iterations > 0 &&
          peer_initialization.trace_iterations > 0,
      "rank2/rank1 initialization discrete parity local(iter=" +
          std::to_string(local_initialization.trace_iterations) +
          ",restart=" +
          std::to_string(local_initialization.trace_restart_dimension) +
          ",max=" +
          std::to_string(local_initialization.trace_maximum_iterations) +
          ") peer(iter=" +
          std::to_string(peer_initialization.trace_iterations) +
          ",restart=" +
          std::to_string(peer_initialization.trace_restart_dimension) +
          ",max=" +
          std::to_string(peer_initialization.trace_maximum_iterations) + ")",
      evidence);
  const std::array<double, 11> local_initialization_scalars = {
      local_initialization.trace_relative_tolerance,
      local_initialization.trace_absolute_tolerance,
      local_initialization.trace_true_residual_norm,
      local_initialization.trace_right_hand_side_norm,
      local_initialization.residual.r_z_norm,
      local_initialization.residual.r_z_scale,
      local_initialization.residual.r_sigma_norm,
      local_initialization.residual.r_sigma_scale,
      local_initialization.residual.g_sigma_norm,
      local_initialization.residual.g_sigma_scale,
      local_initialization.residual.normalized_maximum};
  const std::array<double, 11> peer_initialization_scalars = {
      peer_initialization.trace_relative_tolerance,
      peer_initialization.trace_absolute_tolerance,
      peer_initialization.trace_true_residual_norm,
      peer_initialization.trace_right_hand_side_norm,
      peer_initialization.residual.r_z_norm,
      peer_initialization.residual.r_z_scale,
      peer_initialization.residual.r_sigma_norm,
      peer_initialization.residual.r_sigma_scale,
      peer_initialization.residual.g_sigma_norm,
      peer_initialization.residual.g_sigma_scale,
      peer_initialization.residual.normalized_maximum};
  for (std::size_t scalar = 0;
       scalar < local_initialization_scalars.size(); ++scalar)
  {
    evidence.maximum_operator_relative_error = std::max(
        evidence.maximum_operator_relative_error,
        Rank2RelativeScalar(local_initialization_scalars[scalar],
                            peer_initialization_scalars[scalar]));
  }
  Rank2Require(evidence.maximum_operator_relative_error <= 5.0e-11,
               "rank2/rank1 initialization operator parity", evidence);
  const sc25::Rank2TrajectoryNewtonEvidence& initialization_operator =
      local_run.initialization_operator;
  Rank2Require(initialization_operator.attempted &&
                   initialization_operator.completed &&
                   initialization_operator.iteration == 0,
               "rank2 initialization operator record", evidence);
  Rank2IndependentAssembly initialization_assembly;
  if (AssembleIndependentRank2(
          rank2, rank2_context, initialization_operator.entry,
          initialization_operator.entry, initialization_assembly))
  {
    MakeIndependentAuxiliaryAssembly(initialization_assembly);
    check_operator_vectors(initialization_operator, initialization_assembly,
                           "rank2 initialization");
  }
  else
  {
    Rank2Require(false, "rank2 independent initialization assembly",
                 evidence);
  }
  for (int line = 0; line < sc25::persistent_line_search_trials; ++line)
  {
    Rank2Require(
        initialization_operator.trial_visited[static_cast<std::size_t>(line)] ==
                0 &&
            std::isinf(initialization_operator.trial_residual_norms[
                static_cast<std::size_t>(line)]) &&
            initialization_operator.trial_admissible[
                static_cast<std::size_t>(line)] == 0 &&
            initialization_operator.trial_armijo[
                static_cast<std::size_t>(line)] == 0,
        "rank2 initialization frozen trial sentinel", evidence);
  }

  Rank2Require(checkpoint10.shard_rank == rank &&
                   checkpoint10.descriptor.coordinates.step == 10 &&
                   checkpoint10.descriptor.coordinates.cursor == 10 &&
                   checkpoint10.descriptor.coordinates.generation == 10 &&
                   checkpoint10.descriptor.cursor == 10,
               "rank2 independent checkpoint cadence", evidence);
  evidence.checked_checkpoint_cadence = 1;

  int local_total_newton = 0;
  int local_total_gmres_iterations = 0;
  int local_total_line_trials = 0;
  for (int stage_index = 0; stage_index < sc25::persistent_required_steps;
       ++stage_index)
  {
    const sc25::Rank2TrajectoryStageEvidence& local_stage =
        local_run.stages[static_cast<std::size_t>(stage_index)];
    const sc25::TrajectoryStepEvidence& peer_stage =
        peer_run.steps[static_cast<std::size_t>(stage_index)];
    if (peer_stage.attempt_index < 0 ||
        peer_stage.attempt_index >= peer_run.attempt_count)
    {
      Rank2Require(false, "rank2/rank1 accepted attempt index", evidence);
      continue;
    }
    const sc25::Rank2TrajectoryAttemptEvidence& local_attempt =
        local_run.attempts[static_cast<std::size_t>(stage_index)];
    const sc25::TrajectoryAttemptEvidence& peer_attempt =
        peer_run.attempts[static_cast<std::size_t>(peer_stage.attempt_index)];
    Rank2Require(
        local_stage.accepted && peer_stage.accepted &&
            local_stage.step == peer_stage.step &&
            Rank2RelativeScalar(local_stage.time, peer_stage.time) == 0.0 &&
            local_stage.generation == peer_stage.generation &&
            local_attempt.status == peer_attempt.status &&
            local_attempt.target_step == peer_attempt.target_step &&
            local_attempt.retry_index == peer_attempt.retry_index &&
            local_attempt.newton_iterations ==
                peer_attempt.newton_iterations,
        "rank2/rank1 accepted-stage and attempt bijection stage=" +
            std::to_string(stage_index + 1) + " local(status=" +
            std::to_string(static_cast<int>(local_attempt.status)) +
            ",attempted=" +
            std::to_string(local_attempt.attempted_newton_iterations) +
            ",completed=" +
            std::to_string(local_attempt.newton_iterations) +
            ") peer(status=" +
            std::to_string(static_cast<int>(peer_attempt.status)) +
            ",attempted=" +
            std::to_string(peer_attempt.attempted_newton_iterations) +
            ",completed=" +
            std::to_string(peer_attempt.newton_iterations) + ")",
        evidence);
    const double entry_error = Rank2PeerStateRelativeError(
        rank2, rank1_peer, local_attempt.entry_state,
        peer_attempt.newton[0].entry, evidence, "rank2/rank1 entry state");
    const double stage_error = Rank2PeerStateRelativeError(
        rank2, rank1_peer, local_stage.accepted_state,
        peer_stage.accepted_state, evidence, "rank2/rank1 accepted state");
    evidence.maximum_peer_relative_error = std::max(
        {evidence.maximum_peer_relative_error, entry_error, stage_error});
    Rank2Require(entry_error <= 5.0e-11 && stage_error <= 5.0e-11,
                 "rank2/rank1 state parity", evidence);

    int peer_gmres_iterations = 0;
    int peer_line_trials = 0;
    for (int iteration = 0; iteration < peer_attempt.newton_iterations;
         ++iteration)
    {
      const sc25::TrajectoryNewtonEvidence& peer_newton =
          peer_attempt.newton[static_cast<std::size_t>(iteration)];
      const sc25::Rank2TrajectoryNewtonEvidence& local_newton =
          local_attempt.newton[static_cast<std::size_t>(iteration)];
      peer_gmres_iterations += peer_newton.trace_iterations;
      peer_line_trials += peer_newton.accepted_line_search_index + 1;
      Rank2Require(
          local_newton.attempted && local_newton.completed &&
              local_newton.iteration == iteration &&
              local_newton.trace_iterations >= 0,
          "rank2 completed Newton record stage=" +
              std::to_string(stage_index + 1) + " iteration=" +
              std::to_string(iteration), evidence);
      const double newton_entry_error = Rank2PeerStateRelativeError(
          rank2, rank1_peer, local_newton.entry, peer_newton.entry,
          evidence, "rank2/rank1 Newton entry");
      const double newton_recovery_error = Rank2PeerStateRelativeError(
          rank2, rank1_peer, local_newton.recovered_increment,
          peer_newton.direction, evidence, "rank2/rank1 Newton recovery");
      evidence.maximum_peer_relative_error = std::max(
          {evidence.maximum_peer_relative_error, newton_entry_error,
           newton_recovery_error});
      Rank2Require(newton_entry_error <= 5.0e-11 &&
                       newton_recovery_error <= 5.0e-11,
                   "rank2/rank1 per-Newton state vector parity", evidence);
      Rank2IndependentAssembly independent_assembly;
      if (AssembleIndependentRank2(
              rank2, rank2_context, local_attempt.entry_state,
              local_newton.entry, independent_assembly))
      {
        check_operator_vectors(
            local_newton, independent_assembly,
            "rank2 stage " + std::to_string(stage_index + 1) +
                " Newton " + std::to_string(iteration));
      }
      else
      {
        Rank2Require(false, "rank2 independent per-Newton assembly",
                     evidence);
      }
      for (int line = 0; line < sc25::persistent_line_search_trials; ++line)
      {
        const bool visited =
            line <= local_newton.accepted_line_search_index;
        Rank2Require(
            (local_newton.trial_visited[static_cast<std::size_t>(line)] != 0) ==
                visited,
            "rank2 trial visit bijection", evidence);
        if (visited)
        {
          Rank2Require(
              Rank2RelativeScalar(
                  local_newton.trial_residual_norms[
                      static_cast<std::size_t>(line)],
                  peer_newton.trial_residual_norms[
                      static_cast<std::size_t>(line)]) <= 5.0e-11 &&
                  local_newton.trial_admissible[
                      static_cast<std::size_t>(line)] ==
                      peer_newton.trial_admissible[
                          static_cast<std::size_t>(line)] &&
                  local_newton.trial_armijo[static_cast<std::size_t>(line)] ==
                      peer_newton.trial_armijo[
                          static_cast<std::size_t>(line)],
              "rank2 per-Newton trial scientific parity", evidence);
        }
        else
        {
          Rank2Require(
              std::isinf(local_newton.trial_residual_norms[
                  static_cast<std::size_t>(line)]) &&
                  local_newton.trial_admissible[
                      static_cast<std::size_t>(line)] == 0 &&
                  local_newton.trial_armijo[
                      static_cast<std::size_t>(line)] == 0,
              "rank2 frozen unvisited trial sentinel", evidence);
        }
        ++evidence.checked_trial_records;
      }
      ++evidence.checked_newton_records;
    }
    for (int iteration = local_attempt.attempted_newton_iterations;
         iteration < sc25::persistent_newton_maximum_iterations; ++iteration)
    {
      const sc25::Rank2TrajectoryNewtonEvidence& unvisited =
          local_attempt.newton[static_cast<std::size_t>(iteration)];
      Rank2Require(!unvisited.attempted && !unvisited.completed &&
                       unvisited.iteration == -1 &&
                       std::all_of(
                           unvisited.trial_residual_norms.begin(),
                           unvisited.trial_residual_norms.end(),
                           [](const double value) { return std::isinf(value); }) &&
                       std::all_of(unvisited.trial_visited.begin(),
                                   unvisited.trial_visited.end(),
                                   [](const std::uint8_t value)
                                   { return value == 0; }),
                   "rank2 frozen unvisited Newton sentinel", evidence);
    }
    Rank2Require(local_attempt.gmres_solves ==
                         peer_attempt.newton_iterations &&
                     local_attempt.gmres_iterations > 0 &&
                     peer_gmres_iterations > 0 &&
                     local_attempt.line_search_trials == peer_line_trials,
                 "rank2/rank1 solve and Armijo counter bijection stage=" +
                     std::to_string(stage_index + 1) + " local(gmres=" +
                     std::to_string(local_attempt.gmres_iterations) +
                     ",trials=" +
                     std::to_string(local_attempt.line_search_trials) +
                     ") peer(gmres=" +
                     std::to_string(peer_gmres_iterations) + ",trials=" +
                     std::to_string(peer_line_trials) + ")",
                 evidence);
    if (peer_attempt.newton_iterations > 0)
    {
      const sc25::TrajectoryNewtonEvidence& last_newton =
          peer_attempt.newton[static_cast<std::size_t>(
              peer_attempt.newton_iterations - 1)];
      const double direction_error = Rank2PeerStateRelativeError(
          rank2, rank1_peer, local_attempt.accepted_direction,
          last_newton.direction, evidence, "rank2/rank1 accepted direction");
      evidence.maximum_operator_relative_error = std::max(
          evidence.maximum_operator_relative_error, direction_error);
      Rank2Require(direction_error <= 5.0e-11,
                   "rank2/rank1 recovery direction parity", evidence);
      for (int line = 0; line < sc25::persistent_line_search_trials; ++line)
      {
        const bool visited = line <= last_newton.accepted_line_search_index;
        Rank2Require(
            (visited &&
             Rank2RelativeScalar(
                 local_attempt.trial_residual_norms[
                     static_cast<std::size_t>(line)],
                 last_newton.trial_residual_norms[
                     static_cast<std::size_t>(line)]) <= 5.0e-11 &&
                 local_attempt.trial_admissible[
                     static_cast<std::size_t>(line)] ==
                     last_newton.trial_admissible[
                         static_cast<std::size_t>(line)] &&
                 local_attempt.trial_armijo[
                     static_cast<std::size_t>(line)] ==
                     last_newton.trial_armijo[
                         static_cast<std::size_t>(line)]) ||
                (!visited &&
                 std::isinf(local_attempt.trial_residual_norms[
                     static_cast<std::size_t>(line)]) &&
                 local_attempt.trial_admissible[
                     static_cast<std::size_t>(line)] == 0 &&
                 local_attempt.trial_armijo[
                     static_cast<std::size_t>(line)] == 0),
            "rank2/rank1 final Newton Armijo parity stage=" +
                std::to_string(stage_index + 1) + " line=" +
                std::to_string(line) + " local(residual=" +
                std::to_string(local_attempt.trial_residual_norms[
                    static_cast<std::size_t>(line)]) + ",admissible=" +
                std::to_string(local_attempt.trial_admissible[
                    static_cast<std::size_t>(line)]) + ",armijo=" +
                std::to_string(local_attempt.trial_armijo[
                    static_cast<std::size_t>(line)]) + ") peer(residual=" +
                std::to_string(last_newton.trial_residual_norms[
                    static_cast<std::size_t>(line)]) + ",admissible=" +
                std::to_string(last_newton.trial_admissible[
                    static_cast<std::size_t>(line)]) + ",armijo=" +
                std::to_string(last_newton.trial_armijo[
                    static_cast<std::size_t>(line)]) + ")",
            evidence);
      }
      const std::array<double, 6> local_operator_scalars = {
          local_attempt.entry_residual_norm,
          local_attempt.final_residual_norm,
          local_stage.residual_norm,
          local_stage.condensed_rhs_norm,
          local_stage.trace_solution_norm,
          local_stage.recovery_norm};
      const std::array<double, 6> peer_operator_scalars = {
          peer_attempt.entry_residual_norm,
          peer_attempt.final_residual_norm,
          peer_attempt.final_residual_norm,
          last_newton.trace_right_hand_side_norm,
          PeerStateNorm(last_newton.direction, true),
          PeerStateNorm(last_newton.direction, false)};
      for (std::size_t scalar = 0; scalar < local_operator_scalars.size();
           ++scalar)
      {
        evidence.maximum_operator_relative_error = std::max(
            evidence.maximum_operator_relative_error,
            Rank2RelativeScalar(local_operator_scalars[scalar],
                                peer_operator_scalars[scalar]));
      }
    }
    for (int component = 0; component < sc25::state_components; ++component)
    {
      evidence.maximum_integral_relative_error = std::max({
          evidence.maximum_integral_relative_error,
          Rank2RelativeScalar(
              local_stage.committed_integrals[static_cast<std::size_t>(component)],
              peer_stage.committed_integrals[static_cast<std::size_t>(component)]),
          Rank2RelativeScalar(
              local_stage.conserved_integrals[static_cast<std::size_t>(component)],
              peer_stage.accepted_integrals[static_cast<std::size_t>(component)])});
    }
    local_total_newton += local_attempt.newton_iterations;
    local_total_gmres_iterations += local_attempt.gmres_iterations;
    local_total_line_trials += local_attempt.line_search_trials;
    ++evidence.checked_attempts;
    ++evidence.checked_stages;
  }
  evidence.maximum_operator_relative_error = std::max(
      evidence.maximum_operator_relative_error,
      peer_oracle.maximum_jacobian_action_relative_l2);
  Rank2Require(evidence.checked_restriction_rows == 1280,
               "rank2 independent restriction census", evidence);
  Rank2Require(evidence.checked_owner_bound_face_dofs == 720,
               "rank2 per-face MFEM TDof owner binding census", evidence);
  Rank2Require(evidence.checked_stages == 20 &&
                   evidence.checked_attempts == 20 &&
                   local_run.attempt_count == peer_run.attempt_count &&
                   local_run.accepted_steps == peer_run.accepted_steps &&
                   local_total_newton == peer_oracle.checked_newton_iterations &&
                   evidence.checked_newton_records == local_total_newton &&
                   evidence.checked_trial_records ==
                       sc25::persistent_line_search_trials *
                           local_total_newton &&
                   evidence.checked_vector_records ==
                       6 * (local_total_newton + 1) &&
                   local_total_line_trials == local_total_newton &&
                   peer_oracle.checked_line_search_trials ==
                       sc25::persistent_line_search_trials *
                           peer_oracle.checked_newton_iterations &&
                   local_total_gmres_iterations > 0 &&
                   evidence.independent_rank1_residual_assemblies > 0,
               "rank2 independent stage census", evidence);
  Rank2Require(evidence.maximum_operator_relative_error <= 5.0e-11,
               "rank2/rank1 operator and solver scalar parity", evidence);
  Rank2Require(evidence.maximum_vector_relative_error <= 5.0e-11,
               "rank2 independent vector scientific parity", evidence);
  Rank2Require(evidence.maximum_integral_relative_error <= 5.0e-11,
               "rank2/rank1 integral parity", evidence);
  return evidence.failures.empty();
}

bool RehashIndependentRank2Checkpoint(
    sc25::PersistentRank2TrajectoryCheckpoint& checkpoint) noexcept
{
  checkpoint.raw_field_hashes = IndependentRank2Hashes(checkpoint.state);
  checkpoint.descriptor.interior_hash =
      checkpoint.raw_field_hashes.interior;
  checkpoint.descriptor.trace_hash = checkpoint.raw_field_hashes.trace;
  Rank2ManifestBuilder builder;
  const dg::hdg::runtime::CheckpointDescriptor& descriptor =
      checkpoint.descriptor;
  const dg::hdg::runtime::RuntimeTupleDescriptor& tuple = descriptor.tuple;
  const sc25::Rank2TrajectoryStateHashes& hashes =
      checkpoint.raw_field_hashes;
  const bool valid = builder.Text(sc25::persistent_checkpoint_manifest_id) &&
      builder.Text(checkpoint.projection.view()) &&
      builder.Text(checkpoint.tuple.view()) &&
      builder.Text(checkpoint.scenario.view()) &&
      builder.U64(static_cast<std::uint64_t>(checkpoint.shard_rank)) &&
      builder.U64(descriptor.schema_version) &&
      builder.U64(descriptor.successful_commit ? 1U : 0U) &&
      builder.U64(static_cast<std::uint64_t>(descriptor.restart_mode)) &&
      builder.Text(tuple.model.view()) &&
      builder.Text(tuple.formulation.view()) &&
      builder.Text(tuple.trace.view()) &&
      builder.Text(tuple.stabilization.view()) &&
      builder.Text(tuple.time_treatment.view()) &&
      builder.Text(tuple.backend.view()) &&
      builder.Text(tuple.precision.view()) &&
      builder.Text(descriptor.mfem_pin.view()) &&
      builder.Hash(descriptor.mesh_hash) &&
      builder.Hash(descriptor.partition_hash) &&
      builder.Hash(descriptor.topology_hash) &&
      builder.U64(descriptor.rank_count) &&
      builder.U64(static_cast<std::uint64_t>(descriptor.polynomial_order)) &&
      builder.Text(descriptor.layout.view()) &&
      builder.Text(descriptor.precision.view()) &&
      builder.U64(descriptor.cursor) &&
      builder.U64(dg::hdg::runtime::detail::DoubleBits(descriptor.dt)) &&
      builder.U64(dg::hdg::runtime::detail::DoubleBits(
          descriptor.coordinates.time)) &&
      builder.U64(descriptor.coordinates.step) &&
      builder.U64(descriptor.coordinates.generation) &&
      builder.U64(descriptor.coordinates.cursor) &&
      builder.U64(dg::hdg::runtime::detail::DoubleBits(
          descriptor.coordinates.dt)) &&
      builder.U64(descriptor.coordinate_fingerprint) &&
      builder.Hash(descriptor.interior_hash) &&
      builder.Hash(descriptor.trace_hash) &&
      builder.U64(checkpoint.state.u.size()) &&
      builder.U64(checkpoint.state.sigma.size()) &&
      builder.U64(checkpoint.state.z.size()) &&
      builder.U64(checkpoint.state.uhat.size()) &&
      builder.U64(checkpoint.state.sigmahat.size()) &&
      builder.Hash(hashes.u) && builder.Hash(hashes.sigma) &&
      builder.Hash(hashes.z) && builder.Hash(hashes.uhat) &&
      builder.Hash(hashes.sigmahat) && builder.Hash(hashes.shard);
  const dg::hdg::runtime::StateHash local_manifest = valid ? builder.Finish() :
      dg::hdg::runtime::StateHash{};
  std::array<std::uint8_t, 64> rank_ordered{};
  if (local_manifest == dg::hdg::runtime::StateHash{} ||
      MPI_Allgather(local_manifest.bytes.data(),
                    static_cast<int>(local_manifest.bytes.size()),
                    MPI_UINT8_T, rank_ordered.data(),
                    static_cast<int>(local_manifest.bytes.size()),
                    MPI_UINT8_T, MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    checkpoint.canonical_manifest_hash = {};
    return false;
  }
  Rank2ManifestBuilder combined;
  bool combined_valid =
      combined.Text("HDG-SC25-RANK2-COMBINED-CHECKPOINT-v1") &&
      combined.U64(2) && combined.U64(local_manifest.bytes.size());
  for (int rank = 0; combined_valid && rank < 2; ++rank)
  {
    dg::hdg::runtime::StateHash shard{};
    std::copy_n(rank_ordered.begin() +
                    static_cast<std::ptrdiff_t>(rank) *
                        local_manifest.bytes.size(),
                local_manifest.bytes.size(), shard.bytes.begin());
    combined_valid = combined.U64(static_cast<std::uint64_t>(rank)) &&
        combined.Hash(shard);
  }
  checkpoint.canonical_manifest_hash = combined_valid ? combined.Finish() :
      dg::hdg::runtime::StateHash{};
  return checkpoint.canonical_manifest_hash !=
      dg::hdg::runtime::StateHash{};
}

} // namespace dg_igr_hdg_sc25_cpu_test

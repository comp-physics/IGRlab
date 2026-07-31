#include "sc25_scalar_igr_independent_oracle.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <string>
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
    for (int row = 0; row < c1_true; ++row)
    {
      production_global_schur[static_cast<std::size_t>(row) * c1_true +
                              column] = action[row];
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

} // namespace dg_igr_hdg_sc25_cpu_test

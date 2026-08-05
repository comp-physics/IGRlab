#include "independent_oracle.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace dg_igr_hdg_cpu_test
{

W8cKnownState W8cCellState(const double x) noexcept
{
  constexpr W8cKnownState right{{0.125, 0.0, 0.0, 0.25}};
  constexpr W8cKnownState left{{1.0, 0.0, 0.0, 2.5}};
  return x < 0.25 || x >= 0.75 ? right : left;
}

W8cKnownState W8cInterfaceMean() noexcept
{
  return {{0.5625, 0.0, 0.0, 1.375}};
}

bool VerifyW8cTraceGeometryAndState(
    MPI_Comm communicator, const double* true_trace,
    const std::size_t true_trace_size,
    W8cTraceGeometryEvidence& evidence) noexcept
{
  evidence = {};
  if (communicator == MPI_COMM_NULL || !true_trace) return false;
  int rank = -1;
  int ranks = 0;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS ||
      rank != 0 || ranks != 1)
  {
    return false;
  }

  constexpr int n = 8;
  constexpr int trace_components = 4;
  constexpr int expected_faces = 2 * n * n;
  constexpr int expected_face_nodes = 2 * expected_faces;
  constexpr int expected_true_dofs =
      trace_components * expected_face_nodes;
  constexpr double geometry_tolerance =
      64.0 * std::numeric_limits<double>::epsilon();
  const auto close = [](const double actual, const double expected) noexcept {
    return std::abs(actual - expected) <=
        64.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(actual), std::abs(expected)});
  };
  const auto state_at_x = [](const double x) noexcept {
    return std::abs(x - 0.25) <= geometry_tolerance ||
            std::abs(x - 0.75) <= geometry_tolerance
        ? W8cInterfaceMean() : W8cCellState(x);
  };

  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        n, n, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(2));
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations));
    std::vector<int> partition(
        static_cast<std::size_t>(periodic.GetNE()), 0);
    mfem::ParMesh mesh(communicator, periodic, partition.data());
    mfem::DG_Interface_FECollection collection(
        1, 2, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace space(
        &mesh, &collection, trace_components, mfem::Ordering::byNODES);

    evidence.faces = static_cast<std::uint64_t>(mesh.GetNumFaces());
    evidence.true_dofs = static_cast<std::uint64_t>(space.GetTrueVSize());
    if (mesh.GetNE() != n * n || mesh.GetNumFaces() != expected_faces ||
        space.GetTrueVSize() != expected_true_dofs ||
        true_trace_size != static_cast<std::size_t>(expected_true_dofs))
    {
      return false;
    }

    std::vector<std::uint8_t> assigned(true_trace_size, 0);
    std::vector<int> face_visits(static_cast<std::size_t>(expected_faces), 0);
    for (int face = 0; face < mesh.GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* finite_element = space.GetFaceElement(face);
      mfem::ElementTransformation* transformation =
          mesh.GetFaceTransformation(face);
      mfem::Array<int> vdofs;
      space.GetFaceVDofs(face, vdofs);
      if (!finite_element || !transformation || finite_element->GetDof() != 2 ||
          vdofs.Size() != 2 * trace_components)
      {
        return false;
      }
      const mfem::IntegrationRule& nodes = finite_element->GetNodes();
      if (nodes.GetNPoints() != 2) return false;
      for (int scalar = 0; scalar < 2; ++scalar)
      {
        mfem::Vector point(2);
        transformation->Transform(nodes.IntPoint(scalar), point);
        if (!std::isfinite(point[0]) || !std::isfinite(point[1])) return false;
        const W8cKnownState expected = state_at_x(point[0]);
        ++evidence.face_nodes;
        if (std::abs(point[0] - 0.25) <= geometry_tolerance ||
            std::abs(point[0] - 0.75) <= geometry_tolerance)
        {
          ++evidence.mean_nodes;
        }
        else if (expected == W8cCellState(0.0))
        {
          ++evidence.right_nodes;
        }
        else
        {
          ++evidence.left_nodes;
        }
        if (std::abs(point[0]) <= geometry_tolerance ||
            std::abs(point[0] - 1.0) <= geometry_tolerance)
        {
          ++evidence.periodic_nodes;
        }
        for (int component = 0; component < trace_components; ++component)
        {
          double sign = 1.0;
          const int dof = mfem::FiniteElementSpace::DecodeDof(
              vdofs[scalar + 2 * component], sign);
          const int true_dof = dof >= 0
              ? space.GetLocalTDofNumber(dof) : -1;
          if (dof < 0 || true_dof < 0 ||
              true_dof >= expected_true_dofs || sign == 0.0)
          {
            return false;
          }
          if (sign < 0.0) ++evidence.negative_face_vdofs;
          const double expected_true =
              sign * expected[static_cast<std::size_t>(component)];
          if (!close(true_trace[static_cast<std::size_t>(true_dof)],
                     expected_true) ||
              assigned[static_cast<std::size_t>(true_dof)] != 0)
          {
            return false;
          }
          assigned[static_cast<std::size_t>(true_dof)] = 1;
        }
      }
    }
    if (!std::all_of(assigned.begin(), assigned.end(),
                     [](const std::uint8_t value) { return value == 1; }))
    {
      return false;
    }

    for (int element = 0; element < mesh.GetNE(); ++element)
    {
      mfem::Array<int> faces, orientations;
      mesh.GetElementEdges(element, faces, orientations);
      if (faces.Size() != 4 || orientations.Size() != faces.Size())
        return false;
      for (int local_face = 0; local_face < faces.Size(); ++local_face)
      {
        const int face = faces[local_face];
        const int orientation = orientations[local_face];
        if (face < 0 || face >= expected_faces ||
            (orientation != 1 && orientation != -1))
        {
          return false;
        }
        ++evidence.element_face_visits;
        ++face_visits[static_cast<std::size_t>(face)];
        if (orientation > 0) ++evidence.positive_orientations;
        else ++evidence.negative_orientations;

        const mfem::FiniteElement* finite_element = space.GetFaceElement(face);
        mfem::ElementTransformation* transformation =
            mesh.GetFaceTransformation(face);
        mfem::Array<int> vdofs;
        space.GetFaceVDofs(face, vdofs);
        const int* permutation = collection.DofOrderForOrientation(
            mfem::Geometry::SEGMENT, orientation);
        if (!finite_element || !transformation || !permutation ||
            finite_element->GetDof() != 2 ||
            vdofs.Size() != 2 * trace_components)
        {
          return false;
        }
        std::array<std::uint8_t, 2> permutation_census{};
        const mfem::IntegrationRule& nodes = finite_element->GetNodes();
        for (int oriented = 0; oriented < 2; ++oriented)
        {
          double permutation_sign = 1.0;
          const int canonical = mfem::FiniteElementSpace::DecodeDof(
              permutation[oriented], permutation_sign);
          if (canonical < 0 || canonical >= 2 || permutation_sign == 0.0 ||
              permutation_census[static_cast<std::size_t>(canonical)] != 0)
          {
            return false;
          }
          permutation_census[static_cast<std::size_t>(canonical)] = 1;
          if (permutation_sign < 0.0)
            ++evidence.negative_permutation_dofs;
          mfem::Vector point(2);
          transformation->Transform(nodes.IntPoint(canonical), point);
          const W8cKnownState expected = state_at_x(point[0]);
          for (int component = 0; component < trace_components; ++component)
          {
            double face_sign = 1.0;
            const int dof = mfem::FiniteElementSpace::DecodeDof(
                vdofs[canonical + 2 * component], face_sign);
            const int true_dof = dof >= 0
                ? space.GetLocalTDofNumber(dof) : -1;
            if (true_dof < 0 || true_dof >= expected_true_dofs ||
                !close(face_sign *
                           true_trace[static_cast<std::size_t>(true_dof)],
                       expected[static_cast<std::size_t>(component)]))
            {
              return false;
            }
          }
        }
      }
    }

    const bool exact_face_multiplicity = std::all_of(
        face_visits.begin(), face_visits.end(),
        [](const int visits) { return visits == 2; });
    return exact_face_multiplicity &&
        evidence.faces == expected_faces &&
        evidence.face_nodes == expected_face_nodes &&
        evidence.true_dofs == expected_true_dofs &&
        evidence.element_face_visits == 4 * n * n &&
        evidence.positive_orientations == 2 * n * n &&
        evidence.negative_orientations == 2 * n * n &&
        evidence.negative_face_vdofs == 0 &&
        evidence.negative_permutation_dofs == 0 &&
        evidence.right_nodes == 112 && evidence.left_nodes == 112 &&
        evidence.mean_nodes == 32 && evidence.periodic_nodes == 16;
  }
  catch (...)
  {
    evidence = {};
    return false;
  }
}

namespace
{

constexpr int dim = 2;
constexpr int components = 4;
constexpr double gamma_value = 1.4;
constexpr double tau = 2.5;
constexpr double dt = 0.05;
constexpr double pi = 3.141592653589793238462643383279502884;
using State = std::array<double, components>;
using Jacobian = std::array<std::array<double, components>, components>;
using OracleHash = std::array<std::uint8_t, 32>;

constexpr std::array<std::uint32_t, 64> sha_constants{{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U}};

std::uint32_t RotateRight(const std::uint32_t value,
                          const unsigned int amount) noexcept
{
  return (value >> amount) | (value << (32U - amount));
}

OracleHash OracleHashBytes(const void* data, const std::size_t bytes) noexcept
{
  std::array<std::uint32_t, 8> hash{{0x6a09e667U, 0xbb67ae85U,
                                     0x3c6ef372U, 0xa54ff53aU,
                                     0x510e527fU, 0x9b05688cU,
                                     0x1f83d9abU, 0x5be0cd19U}};
  const auto* input = static_cast<const std::uint8_t*>(data);
  const std::uint64_t bit_count = static_cast<std::uint64_t>(bytes) * 8U;
  const std::size_t blocks = (bytes + 9U + 63U) / 64U;
  std::array<std::uint8_t, 64> block{};
  std::array<std::uint32_t, 64> words{};
  for (std::size_t block_index = 0; block_index < blocks; ++block_index)
  {
    block.fill(0);
    const std::size_t offset = block_index * block.size();
    const std::size_t available = offset < bytes
        ? std::min(block.size(), bytes - offset) : 0;
    if (available != 0) { std::memcpy(block.data(), input + offset, available); }
    if (available < block.size() && offset + available == bytes)
    {
      block[available] = 0x80U;
    }
    if (block_index + 1 == blocks)
    {
      for (unsigned int byte = 0; byte < 8; ++byte)
      {
        block[63U - byte] = static_cast<std::uint8_t>(bit_count >> (8U * byte));
      }
    }
    for (std::size_t index = 0; index < 16; ++index)
    {
      words[index] =
          (static_cast<std::uint32_t>(block[4 * index]) << 24U) |
          (static_cast<std::uint32_t>(block[4 * index + 1]) << 16U) |
          (static_cast<std::uint32_t>(block[4 * index + 2]) << 8U) |
          static_cast<std::uint32_t>(block[4 * index + 3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index)
    {
      const std::uint32_t s0 = RotateRight(words[index - 15], 7) ^
          RotateRight(words[index - 15], 18) ^ (words[index - 15] >> 3U);
      const std::uint32_t s1 = RotateRight(words[index - 2], 17) ^
          RotateRight(words[index - 2], 19) ^ (words[index - 2] >> 10U);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    std::array<std::uint32_t, 8> work = hash;
    for (std::size_t index = 0; index < words.size(); ++index)
    {
      const std::uint32_t s1 = RotateRight(work[4], 6) ^
          RotateRight(work[4], 11) ^ RotateRight(work[4], 25);
      const std::uint32_t choose =
          (work[4] & work[5]) ^ (~work[4] & work[6]);
      const std::uint32_t temporary1 = work[7] + s1 + choose +
          sha_constants[index] + words[index];
      const std::uint32_t s0 = RotateRight(work[0], 2) ^
          RotateRight(work[0], 13) ^ RotateRight(work[0], 22);
      const std::uint32_t majority =
          (work[0] & work[1]) ^ (work[0] & work[2]) ^
          (work[1] & work[2]);
      const std::uint32_t temporary2 = s0 + majority;
      work[7] = work[6];
      work[6] = work[5];
      work[5] = work[4];
      work[4] = work[3] + temporary1;
      work[3] = work[2];
      work[2] = work[1];
      work[1] = work[0];
      work[0] = temporary1 + temporary2;
    }
    for (std::size_t index = 0; index < hash.size(); ++index)
    {
      hash[index] += work[index];
    }
  }
  OracleHash result{};
  for (std::size_t index = 0; index < hash.size(); ++index)
  {
    result[4 * index] = static_cast<std::uint8_t>(hash[index] >> 24U);
    result[4 * index + 1] = static_cast<std::uint8_t>(hash[index] >> 16U);
    result[4 * index + 2] = static_cast<std::uint8_t>(hash[index] >> 8U);
    result[4 * index + 3] = static_cast<std::uint8_t>(hash[index]);
  }
  return result;
}

OracleHash OracleHashDoubles(const std::vector<double>& values) noexcept
{
  return OracleHashBytes(values.data(), values.size() * sizeof(double));
}

OracleHash OracleHashVector(const mfem::Vector& values) noexcept
{
  return OracleHashBytes(values.HostRead(),
                         static_cast<std::size_t>(values.Size()) *
                             sizeof(double));
}

OracleHash OracleHashPair(const OracleHash& left,
                          const OracleHash& right) noexcept
{
  std::array<std::uint8_t, 64> bytes{};
  std::copy(left.begin(), left.end(), bytes.begin());
  std::copy(right.begin(), right.end(), bytes.begin() + 32);
  return OracleHashBytes(bytes.data(), bytes.size());
}

int ByNodes(const int scalar, const int scalar_count,
            const int component) noexcept
{
  return scalar + scalar_count * component;
}

int Decode(const int signed_dof, double& sign) noexcept
{
  return mfem::FiniteElementSpace::DecodeDof(signed_dof, sign);
}

bool Primitive(const State& u, State& q) noexcept
{
  if (!std::all_of(u.begin(), u.end(),
                   [](const double value) { return std::isfinite(value); }) ||
      !(u[0] > 0.0))
  {
    return false;
  }
  const double pressure = (gamma_value - 1.0) *
      (u[3] - 0.5 * (u[1] * u[1] + u[2] * u[2]) / u[0]);
  if (!std::isfinite(pressure) || !(pressure > 0.0)) { return false; }
  q = {{u[0], u[1] / u[0], u[2] / u[0], pressure}};
  return true;
}

bool Conservative(const State& q, State& u) noexcept
{
  if (!(q[0] > 0.0) || !(q[3] > 0.0)) { return false; }
  u = {{q[0], q[0] * q[1], q[0] * q[2],
        q[3] / (gamma_value - 1.0) +
            0.5 * q[0] * (q[1] * q[1] + q[2] * q[2])}};
  return std::all_of(u.begin(), u.end(),
                     [](const double value) { return std::isfinite(value); });
}

State ExactPrimitive(const double x, const double y) noexcept
{
  return {{1.0 + 0.1 * std::sin(2.0 * pi * x) * std::sin(2.0 * pi * y),
           0.35 + 0.05 * std::cos(2.0 * pi * x) * std::sin(2.0 * pi * y),
           -0.20 + 0.05 * std::sin(2.0 * pi * x) * std::cos(2.0 * pi * y),
           1.0 + 0.1 * std::cos(2.0 * pi * x) * std::cos(2.0 * pi * y)}};
}

State Exact(const double x, const double y) noexcept
{
  State result{};
  (void)Conservative(ExactPrimitive(x, y), result);
  return result;
}

State Perturbed(const double x, const double y) noexcept
{
  State q = ExactPrimitive(x, y);
  const double a = std::sin(2.0 * pi * x) * std::cos(2.0 * pi * y);
  const double b = std::cos(2.0 * pi * x) * std::sin(2.0 * pi * y);
  const double c = std::sin(2.0 * pi * (x + y));
  q[0] *= 1.0 + 0.04 * a;
  q[1] += 0.03 * b;
  q[2] -= 0.03 * a;
  q[3] *= 1.0 + 0.04 * c;
  State result{};
  (void)Conservative(q, result);
  return result;
}

bool FluxJacobian(const State& u, const std::array<double, 2>& n,
                  Jacobian& jacobian) noexcept
{
  State qstate{};
  if (!Primitive(u, qstate)) { return false; }
  const double rho = u[0];
  const double mx = u[1];
  const double my = u[2];
  const double normal_momentum = mx * n[0] + my * n[1];
  const double pr = (gamma_value - 1.0) * 0.5 *
      (mx * mx + my * my) / (rho * rho);
  const double px = -(gamma_value - 1.0) * mx / rho;
  const double py = -(gamma_value - 1.0) * my / rho;
  const double pe = gamma_value - 1.0;
  const double enthalpy = u[3] + qstate[3];
  jacobian = {{{{0.0, n[0], n[1], 0.0}},
               {{-mx * normal_momentum / (rho * rho) + pr * n[0],
                 normal_momentum / rho + mx * n[0] / rho + px * n[0],
                 mx * n[1] / rho + py * n[0], pe * n[0]}},
               {{-my * normal_momentum / (rho * rho) + pr * n[1],
                 my * n[0] / rho + px * n[1],
                 normal_momentum / rho + my * n[1] / rho + py * n[1],
                 pe * n[1]}},
               {{pr * normal_momentum / rho -
                     enthalpy * normal_momentum / (rho * rho),
                 px * normal_momentum / rho + enthalpy * n[0] / rho,
                 py * normal_momentum / rho + enthalpy * n[1] / rho,
                 (1.0 + pe) * normal_momentum / rho}}}};
  return true;
}

bool Flux(const State& u, const std::array<double, 2>& n,
          State& flux) noexcept
{
  State q{};
  if (!Primitive(u, q)) { return false; }
  const double normal_momentum = u[1] * n[0] + u[2] * n[1];
  flux = {{normal_momentum,
           u[1] * normal_momentum / u[0] + q[3] * n[0],
           u[2] * normal_momentum / u[0] + q[3] * n[1],
           (u[3] + q[3]) * normal_momentum / u[0]}};
  return true;
}

State ExactDerivative(const double x, const double y,
                      const int direction) noexcept
{
  const double sx = std::sin(2.0 * pi * x);
  const double sy = std::sin(2.0 * pi * y);
  const double cx = std::cos(2.0 * pi * x);
  const double cy = std::cos(2.0 * pi * y);
  const State q = ExactPrimitive(x, y);
  const double dr = direction == 0 ? 0.2 * pi * cx * sy
                                    : 0.2 * pi * sx * cy;
  const double du = direction == 0 ? -0.1 * pi * sx * sy
                                    : 0.1 * pi * cx * cy;
  const double dv = direction == 0 ? 0.1 * pi * cx * cy
                                    : -0.1 * pi * sx * sy;
  const double dp = direction == 0 ? -0.2 * pi * sx * cy
                                    : -0.2 * pi * cx * sy;
  return {{dr, dr * q[1] + q[0] * du, dr * q[2] + q[0] * dv,
           dp / (gamma_value - 1.0) +
               0.5 * dr * (q[1] * q[1] + q[2] * q[2]) +
               q[0] * (q[1] * du + q[2] * dv)}};
}

State Source(const double x, const double y) noexcept
{
  const State state = Exact(x, y);
  Jacobian jx{}, jy{};
  (void)FluxJacobian(state, {{1.0, 0.0}}, jx);
  (void)FluxJacobian(state, {{0.0, 1.0}}, jy);
  const State dx = ExactDerivative(x, y, 0);
  const State dy = ExactDerivative(x, y, 1);
  State source{};
  for (int row = 0; row < components; ++row)
    for (int column = 0; column < components; ++column)
    {
      source[row] += jx[row][column] * dx[column] +
          jy[row][column] * dy[column];
    }
  return source;
}

bool Interpolate(const mfem::Vector& values, const mfem::Array<int>& vdofs,
                 const mfem::Vector& shape, State& state) noexcept
{
  if (vdofs.Size() != shape.Size() * components) { return false; }
  state = {};
  for (int scalar = 0; scalar < shape.Size(); ++scalar)
    for (int component = 0; component < components; ++component)
    {
      double sign = 1.0;
      const int dof = Decode(vdofs[ByNodes(scalar, shape.Size(), component)],
                             sign);
      if (dof < 0 || dof >= values.Size()) { return false; }
      state[component] += shape[scalar] * sign * values[dof];
    }
  return true;
}

struct System
{
  int elements = 0;
  int ni = 0;
  int nt = 0;
  std::vector<double> ri, rt, a, b, c, d, f, g;
  std::vector<int> signed_trace;
};

OracleHash OracleHashSystemMatrix(const System& system) noexcept
{
  OracleHash value = OracleHashDoubles(system.a);
  value = OracleHashPair(value, OracleHashDoubles(system.b));
  value = OracleHashPair(value, OracleHashDoubles(system.c));
  return OracleHashPair(value, OracleHashDoubles(system.d));
}

std::vector<int> Partition(const mfem::Mesh& mesh, const int ranks)
{
  std::vector<int> result(static_cast<std::size_t>(mesh.GetNE()), 0);
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    result[static_cast<std::size_t>(element)] =
        element * ranks / mesh.GetNE();
  }
  return result;
}

bool Assemble(mfem::ParMesh& mesh,
              mfem::ParFiniteElementSpace& interior_space,
              mfem::ParFiniteElementSpace& trace_space,
              const mfem::ParGridFunction& old_interior,
              const mfem::ParGridFunction& current_interior,
              const mfem::ParGridFunction& current_trace,
              const int assembly_quadrature_order,
              System& system)
{
  system = {};
  system.elements = mesh.GetNE();
  const mfem::FiniteElement* first = interior_space.GetFE(0);
  if (!first || first->GetDof() <= 0) { return false; }
  mfem::Array<int> first_faces, first_orientations;
  mesh.GetElementEdges(0, first_faces, first_orientations);
  if (first_faces.Size() != 4) { return false; }
  const mfem::FiniteElement* first_trace =
      trace_space.GetFaceElement(first_faces[0]);
  if (!first_trace || first_trace->GetDof() <= 0) { return false; }
  const int scalar_interior_dofs = first->GetDof();
  const int scalar_trace_dofs = first_trace->GetDof();
  system.ni = first->GetDof() * components;
  system.nt = first_faces.Size() * scalar_trace_dofs * components;
  const std::size_t e = static_cast<std::size_t>(system.elements);
  const std::size_t ni = static_cast<std::size_t>(system.ni);
  const std::size_t nt = static_cast<std::size_t>(system.nt);
  system.ri.assign(e * ni, 0.0);
  system.rt.assign(e * nt, 0.0);
  system.a.assign(e * ni * ni, 0.0);
  system.b.assign(e * ni * nt, 0.0);
  system.c.assign(e * nt * ni, 0.0);
  system.d.assign(e * nt * nt, 0.0);
  system.signed_trace.resize(e * nt);
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    const mfem::FiniteElement* finite_element = interior_space.GetFE(element);
    mfem::ElementTransformation* transformation =
        mesh.GetElementTransformation(element);
    mfem::Array<int> interior_vdofs;
    interior_space.GetElementVDofs(element, interior_vdofs);
    if (!finite_element || !transformation ||
        finite_element->GetDof() != scalar_interior_dofs ||
        interior_vdofs.Size() != system.ni)
    {
      return false;
    }
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(
        finite_element->GetGeomType(), assembly_quadrature_order);
    mfem::Vector shape(scalar_interior_dofs), physical(dim);
    mfem::DenseMatrix derivative(scalar_interior_dofs, dim);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      transformation->Transform(point, physical);
      finite_element->CalcShape(point, shape);
      finite_element->CalcPhysDShape(*transformation, derivative);
      State u{}, uold{}, source = Source(physical[0], physical[1]);
      State fx{}, fy{};
      Jacobian jx{}, jy{};
      if (!Interpolate(current_interior, interior_vdofs, shape, u) ||
          !Interpolate(old_interior, interior_vdofs, shape, uold) ||
          !Flux(u, {{1.0, 0.0}}, fx) || !Flux(u, {{0.0, 1.0}}, fy) ||
          !FluxJacobian(u, {{1.0, 0.0}}, jx) ||
          !FluxJacobian(u, {{0.0, 1.0}}, jy))
      {
        return false;
      }
      const double weight = point.weight * transformation->Weight();
      for (int row = 0; row < scalar_interior_dofs; ++row)
      {
        for (int rc = 0; rc < components; ++rc)
        {
          const int lr = ByNodes(row, scalar_interior_dofs, rc);
          system.ri[static_cast<std::size_t>(element) * ni + lr] += weight *
              (shape[row] * (u[rc] - uold[rc]) / dt -
               derivative(row, 0) * fx[rc] - derivative(row, 1) * fy[rc] -
               shape[row] * source[rc]);
        }
        for (int column = 0; column < scalar_interior_dofs; ++column)
          for (int rc = 0; rc < components; ++rc)
            for (int cc = 0; cc < components; ++cc)
            {
              const int lr = ByNodes(row, scalar_interior_dofs, rc);
              const int lc = ByNodes(column, scalar_interior_dofs, cc);
              const double mass = rc == cc
                  ? shape[row] * shape[column] / dt : 0.0;
              const double volume =
                  -(derivative(row, 0) * jx[rc][cc] +
                    derivative(row, 1) * jy[rc][cc]) * shape[column];
              system.a[static_cast<std::size_t>(element) * ni * ni +
                       static_cast<std::size_t>(lr) * ni + lc] +=
                  weight * (mass + volume);
            }
      }
    }

    mfem::Array<int> faces, orientations;
    mesh.GetElementEdges(element, faces, orientations);
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      mfem::FaceElementTransformations* transformations =
          mesh.GetFaceElementTransformations(face);
      const mfem::FiniteElement* trace_element = trace_space.GetFaceElement(face);
      mfem::Array<int> trace_vdofs;
      trace_space.GetFaceVDofs(face, trace_vdofs);
      if (!transformations || !transformations->Face || !trace_element ||
          trace_element->GetDof() != scalar_trace_dofs ||
          trace_vdofs.Size() != scalar_trace_dofs * components ||
          trace_offset + trace_vdofs.Size() > system.nt)
      {
        return false;
      }
      std::copy(trace_vdofs.begin(), trace_vdofs.end(),
                system.signed_trace.begin() +
                    static_cast<std::size_t>(element) * nt + trace_offset);
      const bool first_side = transformations->Elem1No == element;
      const mfem::IntegrationRule& face_rule =
          mfem::IntRules.Get(trace_element->GetGeomType(),
                             assembly_quadrature_order);
      mfem::Vector ishape(scalar_interior_dofs), tshape(scalar_trace_dofs);
      for (int q = 0; q < face_rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& point = face_rule.IntPoint(q);
        transformations->SetAllIntPoints(&point);
        finite_element->CalcShape(
            first_side ? transformations->GetElement1IntPoint()
                       : transformations->GetElement2IntPoint(), ishape);
        trace_element->CalcShape(point, tshape);
        mfem::Vector scaled_normal(dim);
        mfem::CalcOrtho(transformations->Face->Jacobian(), scaled_normal);
        const double norm = scaled_normal.Norml2();
        const double side = first_side ? 1.0 : -1.0;
        const std::array<double, 2> normal{{side * scaled_normal[0] / norm,
                                            side * scaled_normal[1] / norm}};
        State ui{}, lambda{}, fhat{};
        Jacobian jl{};
        if (!Interpolate(current_interior, interior_vdofs, ishape, ui) ||
            !Interpolate(current_trace, trace_vdofs, tshape, lambda) ||
            !Flux(lambda, normal, fhat) ||
            !FluxJacobian(lambda, normal, jl))
        {
          return false;
        }
        for (int c = 0; c < components; ++c)
        {
          fhat[c] += tau * (ui[c] - lambda[c]);
        }
        const double weight = point.weight * transformations->Face->Weight();
        for (int row = 0; row < scalar_interior_dofs; ++row)
          for (int c = 0; c < components; ++c)
          {
            system.ri[static_cast<std::size_t>(element) * ni +
                      ByNodes(row, scalar_interior_dofs, c)] +=
                weight * ishape[row] * fhat[c];
          }
        for (int row = 0; row < scalar_trace_dofs; ++row)
          for (int c = 0; c < components; ++c)
          {
            system.rt[static_cast<std::size_t>(element) * nt + trace_offset +
                      ByNodes(row, scalar_trace_dofs, c)] +=
                weight * tshape[row] * fhat[c];
          }
        for (int row = 0; row < scalar_interior_dofs; ++row)
          for (int column = 0; column < scalar_interior_dofs; ++column)
            for (int c = 0; c < components; ++c)
            {
              const int lr = ByNodes(row, scalar_interior_dofs, c);
              const int lc = ByNodes(column, scalar_interior_dofs, c);
              system.a[static_cast<std::size_t>(element) * ni * ni +
                       static_cast<std::size_t>(lr) * ni + lc] +=
                  weight * ishape[row] * tau * ishape[column];
            }
        for (int row = 0; row < scalar_interior_dofs; ++row)
          for (int column = 0; column < scalar_trace_dofs; ++column)
            for (int rc = 0; rc < components; ++rc)
              for (int cc = 0; cc < components; ++cc)
              {
                const int lr = ByNodes(row, scalar_interior_dofs, rc);
                const int lc = trace_offset +
                    ByNodes(column, scalar_trace_dofs, cc);
                system.b[static_cast<std::size_t>(element) * ni * nt +
                         static_cast<std::size_t>(lr) * nt + lc] +=
                    weight * ishape[row] * tshape[column] *
                    (jl[rc][cc] - (rc == cc ? tau : 0.0));
              }
        for (int row = 0; row < scalar_trace_dofs; ++row)
          for (int column = 0; column < scalar_interior_dofs; ++column)
            for (int c = 0; c < components; ++c)
            {
              const int lr = trace_offset +
                  ByNodes(row, scalar_trace_dofs, c);
              const int lc = ByNodes(column, scalar_interior_dofs, c);
              system.c[static_cast<std::size_t>(element) * nt * ni +
                       static_cast<std::size_t>(lr) * ni + lc] +=
                  weight * tshape[row] * tau * ishape[column];
            }
        for (int row = 0; row < scalar_trace_dofs; ++row)
          for (int column = 0; column < scalar_trace_dofs; ++column)
            for (int rc = 0; rc < components; ++rc)
              for (int cc = 0; cc < components; ++cc)
              {
                const int lr = trace_offset +
                    ByNodes(row, scalar_trace_dofs, rc);
                const int lc = trace_offset +
                    ByNodes(column, scalar_trace_dofs, cc);
                system.d[static_cast<std::size_t>(element) * nt * nt +
                         static_cast<std::size_t>(lr) * nt + lc] +=
                    weight * tshape[row] * tshape[column] *
                    (jl[rc][cc] - (rc == cc ? tau : 0.0));
              }
      }
      trace_offset += trace_vdofs.Size();
    }
    if (trace_offset != system.nt) { return false; }
  }
  system.f.resize(system.ri.size());
  system.g.resize(system.rt.size());
  std::transform(system.ri.begin(), system.ri.end(), system.f.begin(),
                 [](const double value) { return -value; });
  std::transform(system.rt.begin(), system.rt.end(), system.g.begin(),
                 [](const double value) { return -value; });
  return true;
}

bool Factor(double* matrix, int* pivots, const int n) noexcept
{
  for (int column = 0; column < n; ++column)
  {
    int pivot = column;
    for (int row = column + 1; row < n; ++row)
      if (std::abs(matrix[row * n + column]) >
          std::abs(matrix[pivot * n + column]))
      {
        pivot = row;
      }
    if (!(std::abs(matrix[pivot * n + column]) > 1.0e-14)) { return false; }
    pivots[column] = pivot;
    if (pivot != column)
      for (int j = 0; j < n; ++j)
      {
        std::swap(matrix[column * n + j], matrix[pivot * n + j]);
      }
    for (int row = column + 1; row < n; ++row)
    {
      matrix[row * n + column] /= matrix[column * n + column];
      for (int j = column + 1; j < n; ++j)
      {
        matrix[row * n + j] -=
            matrix[row * n + column] * matrix[column * n + j];
      }
    }
  }
  return true;
}

bool Solve(const double* lu, const int* pivots, const int n,
           double* values, const int right_hand_sides) noexcept
{
  for (int row = 0; row < n; ++row)
    if (pivots[row] != row)
      for (int rhs = 0; rhs < right_hand_sides; ++rhs)
      {
        std::swap(values[row * right_hand_sides + rhs],
                  values[pivots[row] * right_hand_sides + rhs]);
      }
  for (int row = 0; row < n; ++row)
    for (int below = row + 1; below < n; ++below)
      for (int rhs = 0; rhs < right_hand_sides; ++rhs)
      {
        values[below * right_hand_sides + rhs] -=
            lu[below * n + row] * values[row * right_hand_sides + rhs];
      }
  for (int row = n - 1; row >= 0; --row)
    for (int rhs = 0; rhs < right_hand_sides; ++rhs)
    {
      for (int column = row + 1; column < n; ++column)
      {
        values[row * right_hand_sides + rhs] -=
            lu[row * n + column] *
            values[column * right_hand_sides + rhs];
      }
      values[row * right_hand_sides + rhs] /= lu[row * n + row];
    }
  return true;
}

struct Condensed
{
  std::vector<double> lu, b, f, schur, rhs;
  std::vector<int> pivots;
};

bool Condense(const System& system, Condensed& condensed)
{
  const std::size_t elements = static_cast<std::size_t>(system.elements);
  const std::size_t ni = static_cast<std::size_t>(system.ni);
  const std::size_t nt = static_cast<std::size_t>(system.nt);
  condensed.lu = system.a;
  condensed.b = system.b;
  condensed.f = system.f;
  condensed.pivots.resize(elements * ni);
  condensed.schur.resize(elements * nt * nt);
  condensed.rhs.resize(elements * nt);
  std::vector<double> a_inverse_b(ni * nt);
  std::vector<double> a_inverse_f(ni);
  for (int element = 0; element < system.elements; ++element)
  {
    double* lu = condensed.lu.data() +
        static_cast<std::size_t>(element) * ni * ni;
    int* pivots = condensed.pivots.data() +
        static_cast<std::size_t>(element) * ni;
    if (!Factor(lu, pivots, system.ni)) { return false; }
    std::copy(system.b.begin() +
                  static_cast<std::ptrdiff_t>(element * system.ni * system.nt),
              system.b.begin() + static_cast<std::ptrdiff_t>(
                  (element + 1) * system.ni * system.nt),
              a_inverse_b.begin());
    std::copy(system.f.begin() +
                  static_cast<std::ptrdiff_t>(element * system.ni),
              system.f.begin() +
                  static_cast<std::ptrdiff_t>((element + 1) * system.ni),
              a_inverse_f.begin());
    if (!Solve(lu, pivots, system.ni, a_inverse_b.data(), system.nt) ||
        !Solve(lu, pivots, system.ni, a_inverse_f.data(), 1))
    {
      return false;
    }
    for (int row = 0; row < system.nt; ++row)
    {
      double value = system.g[
          static_cast<std::size_t>(element) * nt + row];
      for (int k = 0; k < system.ni; ++k)
      {
        value -= system.c[
            static_cast<std::size_t>(element) * nt * ni +
            static_cast<std::size_t>(row) * ni + k] * a_inverse_f[k];
      }
      condensed.rhs[static_cast<std::size_t>(element) * nt + row] = value;
      for (int column = 0; column < system.nt; ++column)
      {
        value = system.d[
            static_cast<std::size_t>(element) * nt * nt +
            static_cast<std::size_t>(row) * nt + column];
        for (int k = 0; k < system.ni; ++k)
        {
          value -= system.c[
              static_cast<std::size_t>(element) * nt * ni +
              static_cast<std::size_t>(row) * ni + k] *
              a_inverse_b[static_cast<std::size_t>(k) * nt + column];
        }
        condensed.schur[
            static_cast<std::size_t>(element) * nt * nt +
            static_cast<std::size_t>(row) * nt + column] = value;
      }
    }
  }
  return true;
}

double ResidualNorm(MPI_Comm communicator,
                    mfem::ParFiniteElementSpace& trace_space,
                    const System& system)
{
  mfem::Vector local(trace_space.GetVSize());
  mfem::Vector truth(trace_space.GetTrueVSize());
  local = 0.0;
  for (int element = 0; element < system.elements; ++element)
    for (int index = 0; index < system.nt; ++index)
    {
      double sign = 1.0;
      const int dof = Decode(system.signed_trace[
          static_cast<std::size_t>(element) * system.nt + index], sign);
      local[dof] += sign * system.rt[
          static_cast<std::size_t>(element) * system.nt + index];
    }
  trace_space.Dof_TrueDof_Matrix()->MultTranspose(local, truth);
  double local_sum = 0.0;
  for (const double value : system.ri) { local_sum += value * value; }
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, communicator);
  global_sum += mfem::InnerProduct(communicator, truth, truth);
  return std::sqrt(global_sum);
}

bool ProjectTrace(mfem::ParMesh& mesh,
                  mfem::ParFiniteElementSpace& space,
                  mfem::ParGridFunction& local,
                  mfem::Vector& truth,
                  const bool perturbed)
{
  local = 0.0;
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* finite_element = space.GetFaceElement(face);
    mfem::ElementTransformation* transformation =
        mesh.GetFaceTransformation(face);
    mfem::Array<int> vdofs;
    space.GetFaceVDofs(face, vdofs);
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    for (int scalar = 0; scalar < finite_element->GetDof(); ++scalar)
    {
      mfem::Vector physical(dim);
      transformation->Transform(nodes.IntPoint(scalar), physical);
      const State state = perturbed ? Perturbed(physical[0], physical[1])
                                    : Exact(physical[0], physical[1]);
      for (int component = 0; component < components; ++component)
      {
        double sign = 1.0;
        const int dof = Decode(vdofs[ByNodes(
            scalar, finite_element->GetDof(), component)], sign);
        local[dof] = sign * state[component];
      }
    }
  }
  local.ParallelProject(truth);
  local.Distribute(truth);
  return true;
}

bool Admissible(mfem::ParMesh& mesh,
                mfem::ParFiniteElementSpace& interior_space,
                mfem::ParFiniteElementSpace& trace_space,
                const mfem::ParGridFunction& interior,
                const mfem::ParGridFunction& trace,
                const int assembly_quadrature_order)
{
  bool local_ok = true;
  for (int element = 0; element < mesh.GetNE() && local_ok; ++element)
  {
    const mfem::FiniteElement* finite_element = interior_space.GetFE(element);
    mfem::Array<int> vdofs;
    interior_space.GetElementVDofs(element, vdofs);
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(finite_element->GetGeomType(),
                           assembly_quadrature_order);
    mfem::Vector shape(finite_element->GetDof());
    for (int q = 0; q < rule.GetNPoints() && local_ok; ++q)
    {
      finite_element->CalcShape(rule.IntPoint(q), shape);
      State state{}, primitive{};
      local_ok = Interpolate(interior, vdofs, shape, state) &&
          Primitive(state, primitive);
      if (local_ok)
      {
        const double wave = std::hypot(primitive[1], primitive[2]) +
            std::sqrt(gamma_value * primitive[3] / primitive[0]);
        local_ok = primitive[0] >= 1.0e-8 && primitive[3] >= 1.0e-8 &&
            wave < tau;
      }
    }
  }
  for (int face = 0; face < mesh.GetNumFaces() && local_ok; ++face)
  {
    const mfem::FiniteElement* finite_element = trace_space.GetFaceElement(face);
    mfem::Array<int> vdofs;
    trace_space.GetFaceVDofs(face, vdofs);
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(finite_element->GetGeomType(),
                           assembly_quadrature_order);
    mfem::Vector shape(finite_element->GetDof());
    for (int q = 0; q < rule.GetNPoints() && local_ok; ++q)
    {
      finite_element->CalcShape(rule.IntPoint(q), shape);
      State state{}, primitive{};
      local_ok = Interpolate(trace, vdofs, shape, state) &&
          Primitive(state, primitive);
      if (local_ok)
      {
        const double wave = std::hypot(primitive[1], primitive[2]) +
            std::sqrt(gamma_value * primitive[3] / primitive[0]);
        local_ok = primitive[0] >= 1.0e-8 && primitive[3] >= 1.0e-8 &&
            wave < tau;
      }
    }
  }
  int local = local_ok ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, mesh.GetComm());
  return global == 1;
}

} // namespace

bool RunIndependentStep(
    MPI_Comm communicator,
    const IndependentOracleConfig& config,
    const std::uint64_t accepted_step,
    const std::vector<double>& previous_interior,
    const std::vector<double>& previous_true_trace,
    IndependentOracleResult& result)
{
  result = {};
  int ranks = 0;
  MPI_Comm_size(communicator, &ranks);
  const bool functional_mesh = config.elements_per_direction == 3 ||
      config.elements_per_direction == 6 ||
      config.elements_per_direction == 12;
  const bool smoke = config.polynomial_order == 1 &&
      config.elements_per_direction == 3 &&
      config.assembly_quadrature_order == 11;
  const bool functional = config.polynomial_order == 2 &&
      functional_mesh && config.assembly_quadrature_order == 14;
  const bool first_step = accepted_step == 1;
  if ((ranks != 1 && ranks != 2) || (!smoke && !functional) ||
      accepted_step == 0 || accepted_step > 20 ||
      (first_step && (!previous_interior.empty() ||
                      !previous_true_trace.empty())) ||
      (ranks == 2 && config.polynomial_order == 2 &&
       config.elements_per_direction != 6))
  {
    return false;
  }
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        config.elements_per_direction, config.elements_per_direction,
        mfem::Element::QUADRILATERAL, false,
        1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(dim));
    translations[0][0] = 1.0;
    translations[1][1] = 1.0;
    mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations));
    std::vector<int> partition = Partition(periodic, ranks);
    mfem::ParMesh mesh(communicator, periodic, partition.data());
    mfem::L2_FECollection interior_collection(
        config.polynomial_order, dim, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::DG_Interface_FECollection trace_collection(
        config.polynomial_order, dim, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace interior_space(
        &mesh, &interior_collection, components, mfem::Ordering::byNODES);
    mfem::ParFiniteElementSpace trace_space(
        &mesh, &trace_collection, components, mfem::Ordering::byNODES);
    interior_space.ExchangeFaceNbrData();
    trace_space.ExchangeFaceNbrData();
    result.mesh_rebuilt = mesh.GetNE() > 0;

    mfem::ParGridFunction old_interior(&interior_space);
    mfem::ParGridFunction current_interior(&interior_space);
    mfem::ParGridFunction exact_trace(&trace_space);
    mfem::ParGridFunction current_trace(&trace_space);
    mfem::Vector exact_true_trace(trace_space.GetTrueVSize());
    mfem::Vector current_true_trace(trace_space.GetTrueVSize());
    auto exact = [](const mfem::Vector& point, mfem::Vector& values)
    {
      const State state = Exact(point[0], point[1]);
      values.SetSize(components);
      for (int c = 0; c < components; ++c) { values[c] = state[c]; }
    };
    auto perturbed = [](const mfem::Vector& point, mfem::Vector& values)
    {
      const State state = Perturbed(point[0], point[1]);
      values.SetSize(components);
      for (int c = 0; c < components; ++c) { values[c] = state[c]; }
    };
    mfem::VectorFunctionCoefficient exact_coefficient(components, exact);
    mfem::VectorFunctionCoefficient perturbed_coefficient(
        components, perturbed);
    if (first_step)
    {
      old_interior.ProjectCoefficient(exact_coefficient);
      current_interior.ProjectCoefficient(perturbed_coefficient);
      if (!ProjectTrace(mesh, trace_space, exact_trace,
                        exact_true_trace, false) ||
          !ProjectTrace(mesh, trace_space, current_trace,
                        current_true_trace, true))
      {
        return false;
      }
    }
    else
    {
      if (previous_interior.size() !=
              static_cast<std::size_t>(interior_space.GetVSize()) ||
          previous_true_trace.size() !=
              static_cast<std::size_t>(trace_space.GetTrueVSize()))
      {
        return false;
      }
      std::memcpy(old_interior.HostWrite(), previous_interior.data(),
                  previous_interior.size() * sizeof(double));
      std::memcpy(current_interior.HostWrite(), previous_interior.data(),
                  previous_interior.size() * sizeof(double));
      exact_true_trace.SetSize(trace_space.GetTrueVSize());
      current_true_trace.SetSize(trace_space.GetTrueVSize());
      std::memcpy(exact_true_trace.HostWrite(), previous_true_trace.data(),
                  previous_true_trace.size() * sizeof(double));
      std::memcpy(current_true_trace.HostWrite(),
                  previous_true_trace.data(),
                  previous_true_trace.size() * sizeof(double));
      exact_trace.Distribute(exact_true_trace);
      current_trace.Distribute(current_true_trace);
    }
    result.accepted_step = accepted_step;
    result.accepted_time = static_cast<double>(accepted_step) * dt;
    result.initial_interior.assign(
        old_interior.HostRead(), old_interior.HostRead() + old_interior.Size());
    result.initial_true_trace.assign(
        exact_true_trace.HostRead(),
        exact_true_trace.HostRead() + exact_true_trace.Size());

    System system;
    if (!Assemble(mesh, interior_space, trace_space, old_interior,
                  current_interior, current_trace,
                  config.assembly_quadrature_order, system))
    {
      return false;
    }
    result.analytic_source_flux_jacobian = true;
    const bool local_extent_valid = system.signed_trace.size() ==
        static_cast<std::size_t>(system.elements) * system.nt;
    const bool local_entries_valid = local_extent_valid &&
        std::all_of(system.signed_trace.begin(), system.signed_trace.end(),
                    [&](const int signed_dof)
                    {
                      double sign = 1.0;
                      const int local_dof = Decode(signed_dof, sign);
                      return local_dof >= 0 &&
                          local_dof < trace_space.GetVSize() &&
                          (sign == 1.0 || sign == -1.0);
                    });
    std::vector<int> rebuilt_signed_trace(system.signed_trace.size(), 0);
    bool local_permutation_rebuilt = local_extent_valid;
    for (int element = 0;
         element < mesh.GetNE() && local_permutation_rebuilt; ++element)
    {
      mfem::Array<int> faces, orientations;
      mesh.GetElementEdges(element, faces, orientations);
      int offset = 0;
      for (int local_face = 0;
           local_face < faces.Size() && local_permutation_rebuilt; ++local_face)
      {
        mfem::Array<int> face_vdofs;
        trace_space.GetFaceVDofs(faces[local_face], face_vdofs);
        local_permutation_rebuilt = offset + face_vdofs.Size() <= system.nt;
        if (local_permutation_rebuilt)
        {
          std::copy(face_vdofs.begin(), face_vdofs.end(),
                    rebuilt_signed_trace.begin() +
                        static_cast<std::ptrdiff_t>(element * system.nt +
                                                    offset));
          offset += face_vdofs.Size();
        }
      }
      local_permutation_rebuilt = local_permutation_rebuilt &&
          offset == system.nt;
    }
    local_permutation_rebuilt = local_permutation_rebuilt &&
        rebuilt_signed_trace == system.signed_trace;
    const std::uint64_t local_negative_signed_trace_dofs =
        static_cast<std::uint64_t>(std::count_if(
            system.signed_trace.begin(), system.signed_trace.end(),
            [](const int signed_dof) { return signed_dof < 0; }));
    const std::array<int, 3> local_map_flags{{
        local_extent_valid ? 1 : 0,
        local_entries_valid ? 1 : 0,
        local_permutation_rebuilt ? 1 : 0}};
    std::array<int, 3> global_map_flags{};
    if (MPI_Allreduce(local_map_flags.data(), global_map_flags.data(),
                      static_cast<int>(global_map_flags.size()), MPI_INT,
                      MPI_MIN, communicator) != MPI_SUCCESS ||
        MPI_Allreduce(&local_negative_signed_trace_dofs,
                      &result.negative_signed_trace_dofs, 1, MPI_UINT64_T,
                      MPI_SUM, communicator) != MPI_SUCCESS)
    {
      return false;
    }
    result.signed_map_extent_valid = global_map_flags[0] == 1;
    result.signed_map_entries_valid = global_map_flags[1] == 1;
    result.signed_map_permutation_rebuilt = global_map_flags[2] == 1;
    const bool signed_map_structure_valid =
        result.signed_map_extent_valid && result.signed_map_entries_valid &&
        result.signed_map_permutation_rebuilt;
    if (!signed_map_structure_valid) { return false; }
    double residual = ResidualNorm(communicator, trace_space, system);
    result.initial_residual = residual;

    for (int update = 0; update < 20; ++update)
    {
      if (update > 0 && residual <= 1.0e-10 *
                                      (1.0 + result.initial_residual))
      {
        break;
      }
      IndependentLinearizationHashes linearization{};
      IndependentNewtonStepEvidence step{};
      step.update_index = update;
      step.base_residual = residual;
      linearization.residual = OracleHashPair(
          OracleHashDoubles(system.ri), OracleHashDoubles(system.rt));
      Condensed condensed;
      if (!Condense(system, condensed)) { return false; }
      result.local_schur_rebuilt = true;
      linearization.matrix = OracleHashPair(
          OracleHashSystemMatrix(system),
          OracleHashDoubles(condensed.schur));
      mfem::SparseMatrix local_matrix(trace_space.GetVSize());
      mfem::Vector local_rhs(trace_space.GetVSize());
      mfem::Vector true_rhs_values(trace_space.GetTrueVSize());
      local_rhs = 0.0;
      for (int element = 0; element < system.elements; ++element)
      {
        mfem::Array<int> dofs(system.nt);
        mfem::DenseMatrix schur(system.nt);
        for (int row = 0; row < system.nt; ++row)
        {
          const int signed_dof = system.signed_trace[
              static_cast<std::size_t>(element) * system.nt + row];
          dofs[row] = signed_dof;
          double sign = 1.0;
          const int dof = Decode(signed_dof, sign);
          local_rhs[dof] += sign * condensed.rhs[
              static_cast<std::size_t>(element) * system.nt + row];
          for (int column = 0; column < system.nt; ++column)
          {
            schur(row, column) = condensed.schur[
                static_cast<std::size_t>(element) * system.nt * system.nt +
                static_cast<std::size_t>(row) * system.nt + column];
          }
        }
        local_matrix.AddSubMatrix(dofs, dofs, schur);
      }
      local_matrix.Finalize();
      mfem::ParBilinearForm parallel_form(&trace_space);
      std::unique_ptr<mfem::HypreParMatrix> true_matrix(
          parallel_form.ParallelAssemble(&local_matrix));
      trace_space.Dof_TrueDof_Matrix()->MultTranspose(
          local_rhs, true_rhs_values);
      result.global_ptsp_rebuilt = true;

      mfem::HypreParVector true_rhs(&trace_space);
      mfem::HypreParVector true_solution(&trace_space);
      mfem::HypreParVector true_residual(&trace_space);
      std::memcpy(true_rhs.HostWrite(), true_rhs_values.HostRead(),
                  static_cast<std::size_t>(true_rhs.Size()) * sizeof(double));
      true_solution = 0.0;
      mfem::HypreILU ilu;
      ilu.SetType(0);
      ilu.SetLevelOfFill(0);
      ilu.SetMaxIter(1);
      ilu.SetTol(0.0);
      ilu.SetLocalReordering(0);
      ilu.SetPrintLevel(0);
      mfem::HypreGMRES gmres(communicator);
      gmres.SetTol(1.0e-12);
      gmres.SetAbsTol(0.0);
      const int global_trace =
          static_cast<int>(trace_space.GlobalTrueVSize());
      gmres.SetMaxIter(std::max(100, global_trace));
      gmres.SetKDim(std::min(100, std::max(20, global_trace)));
      gmres.SetLogging(1);
      gmres.SetPrintLevel(0);
      gmres.SetPreconditioner(ilu);
      gmres.SetOperator(*true_matrix);
      gmres.Mult(true_rhs, true_solution);
      constexpr std::string_view factor_input_config{
          "hypre-ilu:t0:f0:i1:tol0:r0:p0"};
      linearization.factor = OracleHashPair(
          OracleHashDoubles(condensed.schur),
          OracleHashBytes(factor_input_config.data(),
                          factor_input_config.size()));
      linearization.solution = OracleHashVector(true_solution);
      const double* true_solution_values = true_solution.HostRead();
      step.true_trace_increment.assign(
          true_solution_values,
          true_solution_values + true_solution.Size());
      true_matrix->Mult(true_solution, true_residual);
      true_residual -= true_rhs;
      const double true_relative = std::sqrt(mfem::InnerProduct(
          communicator, true_residual, true_residual)) /
          std::max(std::sqrt(mfem::InnerProduct(
                       communicator, true_rhs, true_rhs)), 1.0e-300);
      result.maximum_true_residual =
          std::max(result.maximum_true_residual, true_relative);
      step.true_relative_residual = true_relative;
      result.true_residual_rebuilt = std::isfinite(true_relative) &&
          true_relative <= 1.0e-10;
      if (!result.true_residual_rebuilt) { return false; }

      mfem::ParGridFunction distributed(&trace_space);
      distributed.Distribute(true_solution);
      result.distribute_rebuilt = true;
      std::vector<double> local_trace_increment(
          static_cast<std::size_t>(system.elements) * system.nt);
      std::vector<double> recovered(
          static_cast<std::size_t>(system.elements) * system.ni);
      for (int element = 0; element < system.elements; ++element)
      {
        std::vector<double> rhs(static_cast<std::size_t>(system.ni));
        for (int local = 0; local < system.nt; ++local)
        {
          double sign = 1.0;
          const int dof = Decode(system.signed_trace[
              static_cast<std::size_t>(element) * system.nt + local], sign);
          local_trace_increment[
              static_cast<std::size_t>(element) * system.nt + local] =
              sign * distributed[dof];
        }
        for (int row = 0; row < system.ni; ++row)
        {
          rhs[static_cast<std::size_t>(row)] = system.f[
              static_cast<std::size_t>(element) * system.ni + row];
          for (int column = 0; column < system.nt; ++column)
          {
            rhs[static_cast<std::size_t>(row)] -= system.b[
                static_cast<std::size_t>(element) * system.ni * system.nt +
                static_cast<std::size_t>(row) * system.nt + column] *
                local_trace_increment[
                    static_cast<std::size_t>(element) * system.nt + column];
          }
        }
        if (!Solve(condensed.lu.data() +
                       static_cast<std::size_t>(element) * system.ni *
                           system.ni,
                   condensed.pivots.data() +
                       static_cast<std::size_t>(element) * system.ni,
                   system.ni, rhs.data(), 1))
        {
          return false;
        }
        std::copy(rhs.begin(), rhs.end(), recovered.begin() +
            static_cast<std::ptrdiff_t>(element * system.ni));
      }
      result.recovery_rebuilt = true;
      std::vector<double> grid_increment(
          static_cast<std::size_t>(interior_space.GetVSize()), 0.0);
      for (int element = 0; element < system.elements; ++element)
      {
        mfem::Array<int> vdofs;
        interior_space.GetElementVDofs(element, vdofs);
        for (int local = 0; local < system.ni; ++local)
        {
          double sign = 1.0;
          const int dof = Decode(vdofs[local], sign);
          grid_increment[static_cast<std::size_t>(dof)] = sign * recovered[
              static_cast<std::size_t>(element) * system.ni + local];
        }
      }

      bool accepted = false;
      for (int trial = 0; trial < 13; ++trial)
      {
        ++step.attempted_trials;
        const double alpha = std::ldexp(1.0, -trial);
        mfem::ParGridFunction candidate_interior(&interior_space);
        mfem::ParGridFunction candidate_trace(&trace_space);
        mfem::Vector candidate_true(trace_space.GetTrueVSize());
        for (int i = 0; i < current_interior.Size(); ++i)
        {
          candidate_interior[i] = current_interior[i] +
              alpha * grid_increment[static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < current_true_trace.Size(); ++i)
        {
          candidate_true[i] = current_true_trace[i] + alpha * true_solution[i];
        }
        candidate_trace.Distribute(candidate_true);
        if (!Admissible(mesh, interior_space, trace_space,
                        candidate_interior, candidate_trace,
                        config.assembly_quadrature_order))
        {
          continue;
        }
        System candidate_system;
        if (!Assemble(mesh, interior_space, trace_space, old_interior,
                      candidate_interior, candidate_trace,
                      config.assembly_quadrature_order, candidate_system))
        {
          return false;
        }
        const double candidate_residual = ResidualNorm(
            communicator, trace_space, candidate_system);
        if (candidate_residual <= (1.0 - 1.0e-4 * alpha) * residual)
        {
          current_interior = candidate_interior;
          current_true_trace = candidate_true;
          current_trace.Distribute(current_true_trace);
          system = std::move(candidate_system);
          residual = candidate_residual;
          linearization.candidate = OracleHashPair(
              OracleHashVector(current_interior),
              OracleHashVector(current_true_trace));
          step.accepted_trial_index = trial;
          step.accepted_alpha = alpha;
          step.accepted_candidate_residual = candidate_residual;
          step.armijo_satisfied = candidate_residual <=
              (1.0 - 1.0e-4 * alpha) * step.base_residual;
          const double* candidate_interior_values = current_interior.HostRead();
          step.accepted_candidate_interior.assign(
              candidate_interior_values,
              candidate_interior_values + current_interior.Size());
          const double* candidate_true_values = current_true_trace.HostRead();
          step.accepted_candidate_true_trace.assign(
              candidate_true_values,
              candidate_true_values + current_true_trace.Size());
          result.linearizations.push_back(linearization);
          result.newton_steps.push_back(step);
          accepted = true;
          break;
        }
      }
      if (!accepted) { return false; }
      result.newton_updates = update + 1;
    }

    result.final_residual = residual;
    result.admissible = Admissible(mesh, interior_space, trace_space,
                                   current_interior, current_trace,
                                   config.assembly_quadrature_order);
    result.interior.assign(current_interior.HostRead(),
                           current_interior.HostRead() +
                               current_interior.Size());
    result.true_trace.assign(current_true_trace.HostRead(),
                             current_true_trace.HostRead() +
                                 current_true_trace.Size());
    const int local_signed_map_used = result.newton_updates > 0 &&
            result.global_ptsp_rebuilt && result.true_residual_rebuilt &&
            result.distribute_rebuilt && result.recovery_rebuilt
        ? 1 : 0;
    int global_signed_map_used = 0;
    if (MPI_Allreduce(&local_signed_map_used, &global_signed_map_used, 1,
                      MPI_INT, MPI_MIN, communicator) != MPI_SUCCESS)
    {
      return false;
    }
    result.signed_map_used = global_signed_map_used == 1;
    result.signed_maps_rebuilt =
        signed_map_structure_valid && result.signed_map_used;
    result.passed = result.mesh_rebuilt &&
        result.analytic_source_flux_jacobian && result.signed_maps_rebuilt &&
        result.local_schur_rebuilt && result.global_ptsp_rebuilt &&
        result.true_residual_rebuilt && result.distribute_rebuilt &&
        result.recovery_rebuilt && result.admissible &&
        result.newton_updates > 0 &&
        result.newton_steps.size() ==
            static_cast<std::size_t>(result.newton_updates) &&
        result.linearizations.size() == result.newton_steps.size() &&
        result.final_residual <= 1.0e-10 * (1.0 + result.initial_residual);
    return result.passed;
  }
  catch (...)
  {
    result.passed = false;
    return false;
  }
}

bool RunIndependentFirstStep(MPI_Comm communicator,
                             const IndependentOracleConfig& config,
                             IndependentOracleResult& result)
{
  return RunIndependentStep(communicator, config, 1, {}, {}, result);
}

bool RunIndependentFirstStep(MPI_Comm communicator,
                             IndependentOracleResult& result)
{
  return RunIndependentFirstStep(
      communicator, IndependentOracleConfig{}, result);
}

} // namespace dg_igr_hdg_cpu_test

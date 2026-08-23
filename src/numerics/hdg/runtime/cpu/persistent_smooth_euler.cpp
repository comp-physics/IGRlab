#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include "hdg_cpu_serial_lifecycle.hpp"

#include "numerics/hdg/core/host_reference.hpp"
#include "numerics/hdg/runtime/runtime_selection.hpp"
#include "numerics/hdg/trace/trace_dg_mpi.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cpu
{

namespace
{

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double density_gate = 1.0e-8;
constexpr double pressure_gate = 1.0e-8;
constexpr std::string_view expected_experiment =
    "exp.hdg.euler.conventional.smooth_persistent.w8a.v1";
constexpr std::string_view expected_source = "source.euler.smooth.analytic.v1";
constexpr std::string_view expected_model = "euler.compressible.2d";
constexpr std::string_view expected_formulation = "hdg.conventional";
constexpr std::string_view expected_trace = "mfem.dg_interface.trace";
constexpr std::string_view expected_stabilization = "tau.fixed";
constexpr std::string_view expected_time = "backward_euler";
constexpr std::string_view expected_precision = "fp64";
constexpr std::string_view expected_mfem_pin =
    "d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec";

constexpr int igr_interior_components = 7;
constexpr int igr_trace_components = 5;

using State = std::array<double, cpu_conserved_components>;
using Matrix4 =
    std::array<std::array<double, cpu_conserved_components>,
               cpu_conserved_components>;
using Matrix45 = std::array<std::array<double, 5>,
                            cpu_conserved_components>;

template <typename T>
StridedBlockView<T> View(T* data, const int batches, const int rows,
                         const int columns) noexcept
{
  return {data, batches, rows, columns, rows * columns, columns, 1};
}

bool Finite(const double value) noexcept { return std::isfinite(value); }

template <std::size_t N>
bool Finite(const std::array<double, N>& values) noexcept
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return Finite(value); });
}

bool Finite(const std::vector<double>& values) noexcept
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return Finite(value); });
}

struct ProjectionDenseLu final
{
  int size = 0;
  std::vector<double> values;
  std::vector<int> pivots;

  bool Factor(const std::vector<double>& matrix, const int n) noexcept
  {
    try
    {
      if (n <= 0 || matrix.size() != static_cast<std::size_t>(n) * n ||
          !Finite(matrix))
      {
        return false;
      }
      size = n;
      values = matrix;
      pivots.assign(static_cast<std::size_t>(n), 0);
      for (int column = 0; column < n; ++column)
      {
        int pivot = column;
        double maximum = std::abs(values[
            static_cast<std::size_t>(column) * n + column]);
        for (int row = column + 1; row < n; ++row)
        {
          const double candidate = std::abs(values[
              static_cast<std::size_t>(row) * n + column]);
          if (candidate > maximum) { maximum = candidate; pivot = row; }
        }
        if (!Finite(maximum) || !(maximum > 0.0)) { return false; }
        pivots[static_cast<std::size_t>(column)] = pivot;
        if (pivot != column)
        {
          for (int entry = 0; entry < n; ++entry)
          {
            std::swap(values[static_cast<std::size_t>(column) * n + entry],
                      values[static_cast<std::size_t>(pivot) * n + entry]);
          }
        }
        const double diagonal = values[
            static_cast<std::size_t>(column) * n + column];
        for (int row = column + 1; row < n; ++row)
        {
          double& multiplier = values[
              static_cast<std::size_t>(row) * n + column];
          multiplier /= diagonal;
          for (int entry = column + 1; entry < n; ++entry)
          {
            values[static_cast<std::size_t>(row) * n + entry] -=
                multiplier * values[
                    static_cast<std::size_t>(column) * n + entry];
          }
        }
      }
      return Finite(values);
    }
    catch (...)
    {
      return false;
    }
  }

  bool Solve(const std::vector<double>& rhs,
             std::vector<double>& solution) const noexcept
  {
    try
    {
      if (size <= 0 || rhs.size() != static_cast<std::size_t>(size) ||
          pivots.size() != rhs.size() || !Finite(rhs))
      {
        return false;
      }
      solution = rhs;
      for (int column = 0; column < size; ++column)
      {
        const int pivot = pivots[static_cast<std::size_t>(column)];
        if (pivot < column || pivot >= size) { return false; }
        if (pivot != column) { std::swap(solution[column], solution[pivot]); }
      }
      for (int column = 0; column < size; ++column)
      {
        for (int row = column + 1; row < size; ++row)
        {
          solution[row] -= values[
              static_cast<std::size_t>(row) * size + column] *
              solution[column];
        }
      }
      for (int row = size - 1; row >= 0; --row)
      {
        for (int column = row + 1; column < size; ++column)
        {
          solution[row] -= values[
              static_cast<std::size_t>(row) * size + column] *
              solution[column];
        }
        const double diagonal = values[
            static_cast<std::size_t>(row) * size + row];
        if (!Finite(diagonal) || diagonal == 0.0) { return false; }
        solution[row] /= diagonal;
      }
      return Finite(solution);
    }
    catch (...)
    {
      return false;
    }
  }
};

StateHash HashBytes(const void*, std::size_t) noexcept;

bool Finite(const mfem::Vector& values) noexcept
{
  const double* data = values.HostRead();
  if (!data && values.Size() != 0) { return false; }
  for (int index = 0; index < values.Size(); ++index)
  {
    if (!Finite(data[index])) { return false; }
  }
  return true;
}

StateHash HashVector(const mfem::Vector& values) noexcept
{
  const double* data = values.HostRead();
  return data || values.Size() == 0
      ? HashBytes(data, static_cast<std::size_t>(values.Size()) *
                            sizeof(double))
      : StateHash{};
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

bool Primitive(const State& conservative, State& primitive) noexcept
{
  if (!Finite(conservative) || !(conservative[0] > 0.0)) { return false; }
  const double rho = conservative[0];
  const double kinetic = 0.5 *
      (conservative[1] * conservative[1] +
       conservative[2] * conservative[2]) / rho;
  const double pressure =
      (cpu_gamma - 1.0) * (conservative[3] - kinetic);
  if (!Finite(pressure) || !(pressure > 0.0)) { return false; }
  primitive = {{rho, conservative[1] / rho, conservative[2] / rho,
                pressure}};
  return Finite(primitive);
}

bool Conservative(const State& primitive, State& conservative) noexcept
{
  if (!Finite(primitive) || !(primitive[0] > 0.0) ||
      !(primitive[3] > 0.0))
  {
    return false;
  }
  const double rho = primitive[0];
  const double u = primitive[1];
  const double v = primitive[2];
  conservative = {{rho, rho * u, rho * v,
                   primitive[3] / (cpu_gamma - 1.0) +
                       0.5 * rho * (u * u + v * v)}};
  return Finite(conservative);
}

bool ManufacturedPrimitive(const double x, const double y,
                           State& primitive) noexcept
{
  if (!Finite(x) || !Finite(y)) { return false; }
  primitive = {{1.0 + 0.1 * std::sin(2.0 * pi * x) *
                          std::sin(2.0 * pi * y),
                0.35 + 0.05 * std::cos(2.0 * pi * x) *
                           std::sin(2.0 * pi * y),
                -0.20 + 0.05 * std::sin(2.0 * pi * x) *
                            std::cos(2.0 * pi * y),
                1.0 + 0.1 * std::cos(2.0 * pi * x) *
                          std::cos(2.0 * pi * y)}};
  return true;
}

bool ManufacturedConservative(const double x, const double y,
                              State& conservative) noexcept
{
  State primitive{};
  return ManufacturedPrimitive(x, y, primitive) &&
      Conservative(primitive, conservative);
}

bool W8cConservative(const double x, State& conservative) noexcept
{
  if (!Finite(x)) { return false; }
  conservative = x < 0.25 || x >= 0.75 ? w8c::right_state
                                        : w8c::left_state;
  return true;
}

bool W8cTraceConservative(const double x, State& conservative) noexcept
{
  if (!Finite(x)) { return false; }
  constexpr double tolerance =
      64.0 * std::numeric_limits<double>::epsilon();
  if (std::abs(x - 0.25) <= tolerance ||
      std::abs(x - 0.75) <= tolerance)
  {
    conservative = w8c::interface_mean;
    return true;
  }
  return W8cConservative(x, conservative);
}

bool SmoothTransitionConservative(const double x,
                                  State& conservative) noexcept
{
  if (!Finite(x)) { return false; }
  const double kappa = std::atanh(0.8) /
      std::sin(pi * w8c::SmoothTransitionV2Case::transition_width);
  const double chi = 0.5 *
      (1.0 - std::tanh(kappa * std::cos(2.0 * pi * x)));
  const double rho = 0.125 + 0.875 * chi;
  const double pressure = 0.1 + 0.9 * chi;
  conservative = {{rho, 0.0, 0.0, pressure / (cpu_gamma - 1.0)}};
  return Finite(conservative);
}

State ManufacturedDerivative(const double x, const double y,
                             const int direction) noexcept
{
  const double sx = std::sin(2.0 * pi * x);
  const double sy = std::sin(2.0 * pi * y);
  const double cx = std::cos(2.0 * pi * x);
  const double cy = std::cos(2.0 * pi * y);
  State primitive{};
  (void)ManufacturedPrimitive(x, y, primitive);
  const double drho = direction == 0 ? 0.2 * pi * cx * sy
                                      : 0.2 * pi * sx * cy;
  const double du = direction == 0 ? -0.1 * pi * sx * sy
                                    : 0.1 * pi * cx * cy;
  const double dv = direction == 0 ? 0.1 * pi * cx * cy
                                    : -0.1 * pi * sx * sy;
  const double dp = direction == 0 ? -0.2 * pi * sx * cy
                                    : -0.2 * pi * cx * sy;
  const double rho = primitive[0];
  const double u = primitive[1];
  const double v = primitive[2];
  return {{drho, drho * u + rho * du, drho * v + rho * dv,
           dp / (cpu_gamma - 1.0) +
               0.5 * drho * (u * u + v * v) +
               rho * (u * du + v * dv)}};
}

bool NormalFlux(const State& conservative,
                const std::array<double, 2>& normal,
                State& flux) noexcept
{
  State primitive{};
  if (!Primitive(conservative, primitive) || !Finite(normal) ||
      (normal[0] == 0.0 && normal[1] == 0.0))
  {
    return false;
  }
  const double rho = primitive[0];
  const double pressure = primitive[3];
  const double normal_momentum =
      conservative[1] * normal[0] + conservative[2] * normal[1];
  flux = {{normal_momentum,
           conservative[1] * normal_momentum / rho + pressure * normal[0],
           conservative[2] * normal_momentum / rho + pressure * normal[1],
           (conservative[3] + pressure) * normal_momentum / rho}};
  return Finite(flux);
}

bool NormalFluxJacobian(const State& conservative,
                        const std::array<double, 2>& normal,
                        Matrix4& jacobian) noexcept
{
  State primitive{};
  if (!Primitive(conservative, primitive) || !Finite(normal) ||
      (normal[0] == 0.0 && normal[1] == 0.0))
  {
    return false;
  }
  const double rho = conservative[0];
  const double mx = conservative[1];
  const double my = conservative[2];
  const double energy = conservative[3];
  const double pressure = primitive[3];
  const double q = mx * normal[0] + my * normal[1];
  const double p_rho = (cpu_gamma - 1.0) * 0.5 *
      (mx * mx + my * my) / (rho * rho);
  const double p_mx = -(cpu_gamma - 1.0) * mx / rho;
  const double p_my = -(cpu_gamma - 1.0) * my / rho;
  const double p_energy = cpu_gamma - 1.0;
  const double enthalpy = energy + pressure;
  jacobian = {{{{0.0, normal[0], normal[1], 0.0}},
               {{-mx * q / (rho * rho) + p_rho * normal[0],
                 q / rho + mx * normal[0] / rho + p_mx * normal[0],
                 mx * normal[1] / rho + p_my * normal[0],
                 p_energy * normal[0]}},
               {{-my * q / (rho * rho) + p_rho * normal[1],
                 my * normal[0] / rho + p_mx * normal[1],
                 q / rho + my * normal[1] / rho + p_my * normal[1],
                 p_energy * normal[1]}},
               {{p_rho * q / rho - enthalpy * q / (rho * rho),
                 p_mx * q / rho + enthalpy * normal[0] / rho,
                 p_my * q / rho + enthalpy * normal[1] / rho,
                 (1.0 + p_energy) * q / rho}}}};
  return std::all_of(jacobian.begin(), jacobian.end(),
                     [](const auto& row) { return Finite(row); });
}

bool RegularizedNormalFlux(const State& conservative,
                           const double sigma,
                           const std::array<double, 2>& normal,
                           State& flux,
                           Matrix45& jacobian) noexcept
{
  Matrix4 euler_jacobian{};
  if (!Finite(sigma) || !NormalFlux(conservative, normal, flux) ||
      !NormalFluxJacobian(conservative, normal, euler_jacobian) ||
      !(conservative[0] > 0.0))
  {
    return false;
  }
  for (int row = 0; row < cpu_conserved_components; ++row)
  {
    for (int column = 0; column < cpu_conserved_components; ++column)
      jacobian[row][column] = euler_jacobian[row][column];
    jacobian[row][4] = 0.0;
  }
  const double normal_velocity =
      (conservative[1] * normal[0] + conservative[2] * normal[1]) /
      conservative[0];
  flux[1] += sigma * normal[0];
  flux[2] += sigma * normal[1];
  flux[3] += sigma * normal_velocity;
  jacobian[1][4] = normal[0];
  jacobian[2][4] = normal[1];
  jacobian[3][0] -= sigma * normal_velocity / conservative[0];
  jacobian[3][1] += sigma * normal[0] / conservative[0];
  jacobian[3][2] += sigma * normal[1] / conservative[0];
  jacobian[3][4] = normal_velocity;
  return Finite(flux) &&
      std::all_of(jacobian.begin(), jacobian.end(),
                  [](const auto& row) { return Finite(row); });
}

bool NormalSpectralRadius(const State& conservative,
                          const std::array<double, 2>& normal,
                          double& radius) noexcept
{
  State primitive{};
  if (!Primitive(conservative, primitive) || !Finite(normal) ||
      (normal[0] == 0.0 && normal[1] == 0.0))
  {
    return false;
  }
  const double normal_velocity =
      primitive[1] * normal[0] + primitive[2] * normal[1];
  const double sound_speed =
      std::sqrt(cpu_gamma * primitive[3] / primitive[0]);
  radius = std::abs(normal_velocity) + sound_speed;
  return Finite(radius) && radius > 0.0;
}

bool BorrowedEulerLlfStabilization(
    const State& interior, const State& trace,
    const std::array<double, 2>& normal, double& stabilization) noexcept
{
  double interior_radius = 0.0;
  double trace_radius = 0.0;
  if (!NormalSpectralRadius(interior, normal, interior_radius) ||
      !NormalSpectralRadius(trace, normal, trace_radius))
  {
    return false;
  }
  stabilization = std::max(interior_radius, trace_radius);
  return Finite(stabilization) && stabilization > 0.0;
}

bool ManufacturedSource(const double x, const double y,
                        State& source) noexcept
{
  State state{};
  Matrix4 jacobian_x{}, jacobian_y{};
  if (!ManufacturedConservative(x, y, state) ||
      !NormalFluxJacobian(state, {{1.0, 0.0}}, jacobian_x) ||
      !NormalFluxJacobian(state, {{0.0, 1.0}}, jacobian_y))
  {
    return false;
  }
  const State dx = ManufacturedDerivative(x, y, 0);
  const State dy = ManufacturedDerivative(x, y, 1);
  source = {};
  for (int row = 0; row < cpu_conserved_components; ++row)
    for (int column = 0; column < cpu_conserved_components; ++column)
    {
      source[row] += jacobian_x[row][column] * dx[column] +
          jacobian_y[row][column] * dy[column];
    }
  return Finite(source);
}

bool PerturbedConservative(const double x, const double y,
                          State& conservative) noexcept
{
  State primitive{};
  if (!ManufacturedPrimitive(x, y, primitive)) { return false; }
  const double psi_one =
      std::sin(2.0 * pi * x) * std::cos(2.0 * pi * y);
  const double psi_two =
      std::cos(2.0 * pi * x) * std::sin(2.0 * pi * y);
  const double psi_three = std::sin(2.0 * pi * (x + y));
  primitive[0] *= 1.0 + 0.04 * psi_one;
  primitive[1] += 0.03 * psi_two;
  primitive[2] -= 0.03 * psi_one;
  primitive[3] *= 1.0 + 0.04 * psi_three;
  return Conservative(primitive, conservative);
}

bool Interpolate(const mfem::Vector& field,
                 const int* signed_vdofs,
                 const int scalar_dofs,
                 const mfem::Vector& shape,
                 State& state) noexcept
{
  if (signed_vdofs == nullptr || shape.Size() != scalar_dofs) { return false; }
  const double* values = field.HostRead();
  if (values == nullptr) { return false; }
  state = {};
  for (int scalar = 0; scalar < scalar_dofs; ++scalar)
    for (int component = 0; component < cpu_conserved_components; ++component)
    {
      double sign = 1.0;
      const int dof = Decode(
          signed_vdofs[ByNodes(scalar, scalar_dofs, component)], sign);
      if (dof < 0 || dof >= field.Size()) { return false; }
      state[component] += shape[scalar] * sign * values[dof];
    }
  return Finite(state);
}

bool InterpolateComponent(const mfem::Vector& field,
                          const int* signed_vdofs,
                          const int scalar_dofs,
                          const mfem::Vector& shape,
                          const int component,
                          double& value) noexcept
{
  if (signed_vdofs == nullptr || shape.Size() != scalar_dofs ||
      component < 0)
  {
    return false;
  }
  const double* values = field.HostRead();
  if (values == nullptr) { return false; }
  value = 0.0;
  for (int scalar = 0; scalar < scalar_dofs; ++scalar)
  {
    double sign = 1.0;
    const int dof = Decode(
        signed_vdofs[ByNodes(scalar, scalar_dofs, component)], sign);
    if (dof < 0 || dof >= field.Size()) { return false; }
    value += shape[scalar] * sign * values[dof];
  }
  return Finite(value);
}

bool InterpolateDerivative(const mfem::Vector& field,
                           const int* signed_vdofs,
                           const int scalar_dofs,
                           const mfem::DenseMatrix& derivative,
                           const int component,
                           const int direction,
                           double& value) noexcept
{
  if (signed_vdofs == nullptr || derivative.Height() != scalar_dofs ||
      direction < 0 || direction >= derivative.Width() || component < 0)
  {
    return false;
  }
  const double* values = field.HostRead();
  if (values == nullptr) { return false; }
  value = 0.0;
  for (int scalar = 0; scalar < scalar_dofs; ++scalar)
  {
    double sign = 1.0;
    const int dof = Decode(
        signed_vdofs[ByNodes(scalar, scalar_dofs, component)], sign);
    if (dof < 0 || dof >= field.Size()) { return false; }
    value += derivative(scalar, direction) * sign * values[dof];
  }
  return Finite(value);
}

// Compact SHA-256 used for committed bytes and product payload identities.
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

StateHash HashBytes(const void* data, const std::size_t bytes) noexcept
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
  StateHash result{};
  for (std::size_t index = 0; index < hash.size(); ++index)
  {
    result.bytes[4 * index] = static_cast<std::uint8_t>(hash[index] >> 24U);
    result.bytes[4 * index + 1] =
        static_cast<std::uint8_t>(hash[index] >> 16U);
    result.bytes[4 * index + 2] =
        static_cast<std::uint8_t>(hash[index] >> 8U);
    result.bytes[4 * index + 3] = static_cast<std::uint8_t>(hash[index]);
  }
  return result;
}

StateHash HashDoubles(const std::vector<double>& values) noexcept
{
  return HashBytes(values.data(), values.size() * sizeof(double));
}

StateHash HashPair(const StateHash& left, const StateHash& right) noexcept
{
  std::array<std::uint8_t, 64> bytes{};
  std::copy(left.bytes.begin(), left.bytes.end(), bytes.begin());
  std::copy(right.bytes.begin(), right.bytes.end(), bytes.begin() + 32);
  return HashBytes(bytes.data(), bytes.size());
}

StateHash RankCommonHash(const std::vector<double>& values) noexcept
{
  int ranks = 0;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS ||
      (ranks != 1 && ranks != 2))
  {
    return {};
  }
  std::array<std::uint8_t, 40> local{};
  const std::uint64_t size = static_cast<std::uint64_t>(values.size());
  for (unsigned int byte = 0; byte < 8; ++byte)
  {
    local[byte] = static_cast<std::uint8_t>(size >> (8U * byte));
  }
  const StateHash local_hash = HashDoubles(values);
  std::copy(local_hash.bytes.begin(), local_hash.bytes.end(),
            local.begin() + 8);
  std::array<std::array<std::uint8_t, 40>, 2> rank_ordered{};
  if (MPI_Allgather(local.data(), static_cast<int>(local.size()), MPI_BYTE,
                    rank_ordered.data(), static_cast<int>(local.size()),
                    MPI_BYTE, MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return {};
  }
  return HashBytes(rank_ordered.data(),
                   static_cast<std::size_t>(ranks) * local.size());
}

std::uint64_t FailureMask(const FailureReason reason) noexcept
{
  const unsigned int value = static_cast<unsigned int>(reason);
  return value > 0 && value < static_cast<unsigned int>(FailureReason::count)
      ? (std::uint64_t{1} << value) : 0;
}

std::uint64_t CompatibilityMask(
    const RestartCompatibility compatibility) noexcept
{
  const unsigned int value = static_cast<unsigned int>(compatibility);
  return value > 0 &&
          value < static_cast<unsigned int>(RestartCompatibility::count)
      ? (std::uint64_t{1} << value) : 0;
}

FailureReason FirstFailure(const std::uint64_t mask) noexcept
{
  for (unsigned int value = 1;
       value < static_cast<unsigned int>(FailureReason::count); ++value)
  {
    if ((mask & (std::uint64_t{1} << value)) != 0)
    {
      return static_cast<FailureReason>(value);
    }
  }
  return FailureReason::none;
}

RestartCompatibility FirstCompatibility(const std::uint64_t mask) noexcept
{
  for (unsigned int value = 1;
       value < static_cast<unsigned int>(RestartCompatibility::count); ++value)
  {
    if ((mask & (std::uint64_t{1} << value)) != 0)
    {
      return static_cast<RestartCompatibility>(value);
    }
  }
  return RestartCompatibility::compatible;
}

template <typename Context, typename Same>
bool ContextsEqual(const Context& context, Same&& same,
                   int& ranks) noexcept
{
  static_assert(std::is_trivially_copyable<Context>::value,
                "collective contexts must be trivially copyable");
  ranks = 0;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS ||
      (ranks != 1 && ranks != 2))
  {
    return false;
  }
  std::array<Context, 2> gathered{};
  if (MPI_Allgather(&context, static_cast<int>(sizeof(Context)), MPI_BYTE,
                    gathered.data(), static_cast<int>(sizeof(Context)),
                    MPI_BYTE, MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  for (int rank = 0; rank < ranks; ++rank)
  {
    if (!same(context, gathered[static_cast<std::size_t>(rank)]))
    {
      return false;
    }
  }
  return true;
}

bool AllRanks(const bool local) noexcept
{
  int value = local ? 1 : 0;
  int reduced = 0;
  return MPI_Allreduce(&value, &reduced, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && reduced == 1;
}

struct CpuCollectiveTestLogicalState
{
  detail::CpuCollectiveTestFault armed_fault =
      detail::CpuCollectiveTestFault::none;
  detail::CpuCollectiveTestFault consumed_fault =
      detail::CpuCollectiveTestFault::none;
  std::uint64_t arm_generation = 0;
  std::uint64_t local_consumption_count = 0;
  std::uint64_t owner_incarnation = 0;
  std::uint64_t consumed_incarnation = 0;
  std::uint64_t last_rejected_incarnation = 0;
  std::uint64_t local_wrong_owner_rejections = 0;
  std::uint64_t local_reset_completions = 0;
  int target_rank = -1;
  int consumed_rank = -1;
  bool armed = false;
  bool reset_collectively_attested = false;
};

// Arming, execution, and observation are deliberately confined to the same
// test thread.  An unrelated adapter running on another thread cannot observe
// or consume this private fault state.
thread_local CpuCollectiveTestLogicalState collective_test_state{};

struct CpuGenericBoundaryPicardTestState
{
  std::uint64_t callbacks = 0;
  std::uint64_t physical_rows = 0;
  std::uint64_t zero_c_rows = 0;
  std::uint64_t nonzero_d_rows = 0;
};

thread_local CpuGenericBoundaryPicardTestState
    generic_boundary_picard_test_state{};

bool ConsumeOwnedCollectiveTestFault(
    const detail::CpuCollectiveTestFault expected, const int rank,
    const std::uint64_t incarnation) noexcept
{
  if (!collective_test_state.armed ||
      collective_test_state.armed_fault != expected)
  {
    return false;
  }
  if (collective_test_state.owner_incarnation != incarnation)
  {
    ++collective_test_state.local_wrong_owner_rejections;
    collective_test_state.last_rejected_incarnation = incarnation;
    return false;
  }
  collective_test_state.armed = false;
  collective_test_state.armed_fault = detail::CpuCollectiveTestFault::none;
  if (collective_test_state.target_rank != rank) { return false; }
  ++collective_test_state.local_consumption_count;
  collective_test_state.consumed_incarnation = incarnation;
  collective_test_state.consumed_fault = expected;
  collective_test_state.consumed_rank = rank;
  return true;
}

void RecordCollectiveTestReset(const std::uint64_t incarnation,
                               const bool collectively_attested) noexcept
{
  if (collective_test_state.owner_incarnation != incarnation) { return; }
  if (collectively_attested)
  {
    ++collective_test_state.local_reset_completions;
  }
  collective_test_state.reset_collectively_attested = collectively_attested;
}

bool RefuseCollectiveTestReset(const std::uint64_t incarnation,
                               const int rank) noexcept
{
  return collective_test_state.owner_incarnation == incarnation &&
      collective_test_state.consumed_fault ==
          detail::CpuCollectiveTestFault::
              initialization_facade_readiness_reset_refusal &&
      collective_test_state.target_rank == rank;
}

std::uint64_t NextCoreIncarnation() noexcept
{
  static std::atomic<std::uint64_t> next{1};
  const std::uint64_t value = next.fetch_add(1, std::memory_order_relaxed);
  return value == 0 ? 0 : value;
}

struct CpuCheckpointImportAdmission
{
  CheckpointDescriptor descriptor{};
  std::uint32_t representation_version = 1;
  int local_ready = 0;
};
static_assert(std::is_trivially_copyable<
                  CpuCheckpointImportAdmission>::value,
              "checkpoint import admission has a fixed MPI representation");

bool AdmitCpuCheckpointImport(const CheckpointDescriptor& descriptor,
                              const bool local_ready) noexcept
{
  const CpuCheckpointImportAdmission admission{
      descriptor, 1, local_ready ? 1 : 0};
  int ranks = 0;
  const bool same = ContextsEqual(
      admission,
      [](const CpuCheckpointImportAdmission& left,
         const CpuCheckpointImportAdmission& right)
      {
        return left.representation_version == 1 &&
            right.representation_version == 1 && left.local_ready == 1 &&
            right.local_ready == 1 &&
            ::dg::hdg::runtime::detail::SameCheckpoint(
                left.descriptor, right.descriptor);
      },
      ranks);
  return same && descriptor.rank_count == static_cast<std::uint32_t>(ranks);
}

std::uint64_t SumU64(const std::uint64_t local,
                     const MPI_Comm communicator = MPI_COMM_WORLD) noexcept
{
  unsigned long long input = static_cast<unsigned long long>(local);
  unsigned long long output = 0;
  if (MPI_Allreduce(&input, &output, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                    communicator) != MPI_SUCCESS)
  {
    return 0;
  }
  return static_cast<std::uint64_t>(output);
}

} // namespace

void detail::ArmCpuCollectiveTestFault(
    const detail::CpuCollectiveTestFault fault, const int target_rank,
    const std::uint64_t owner_incarnation) noexcept
{
  const std::uint64_t next_generation = collective_test_state.arm_generation + 1;
  collective_test_state = {};
  collective_test_state.arm_generation =
      next_generation == 0 ? 1 : next_generation;
  collective_test_state.armed_fault = fault;
  collective_test_state.owner_incarnation = owner_incarnation;
  collective_test_state.target_rank = target_rank;
  collective_test_state.armed = owner_incarnation != 0 &&
      fault != detail::CpuCollectiveTestFault::none;
}

void detail::ResetCpuCollectiveTestFault() noexcept
{
  collective_test_state = {};
}

detail::CpuCollectiveTestConsumptionEvidence
detail::ObserveCpuCollectiveTestFault() noexcept
{
  return {collective_test_state.arm_generation,
          collective_test_state.local_consumption_count,
          collective_test_state.owner_incarnation,
          collective_test_state.consumed_incarnation,
          collective_test_state.last_rejected_incarnation,
          collective_test_state.local_wrong_owner_rejections,
          collective_test_state.local_reset_completions,
          collective_test_state.consumed_fault,
          collective_test_state.consumed_rank,
          collective_test_state.target_rank,
          collective_test_state.armed,
          collective_test_state.reset_collectively_attested,
          true};
}

void detail::ResetCpuGenericBoundaryPicardTestObservation() noexcept
{
  generic_boundary_picard_test_state = {};
}

bool detail::CpuGenericBoundaryPicardAssemblyObservedForTests() noexcept
{
  const CpuGenericBoundaryPicardTestState& state =
      generic_boundary_picard_test_state;
  return state.callbacks != 0 && state.callbacks == state.physical_rows &&
      state.zero_c_rows == state.physical_rows &&
      state.nonzero_d_rows == state.physical_rows;
}

struct CpuSlotStorage
{
  std::vector<double> interior;
  std::vector<double> trace;
};

struct CpuSlotCell
{
  std::unique_ptr<CpuSlotStorage> backing;
};

namespace
{

struct CpuOwnedAllocationLedger
{
  std::array<CpuOwnedAllocationEvent, 96> events{};
  std::size_t count = 0;
  std::size_t baseline = 0;
  bool sealed = false;

  bool Observe(const CpuOwnedAllocationSite site,
               const std::size_t bytes,
               const std::size_t alignment,
               const std::size_t capacity,
               const void* token) noexcept
  {
    if (!token || bytes == 0 || alignment == 0 || count >= events.size())
    {
      return false;
    }
    events[count] = {site, static_cast<std::uint64_t>(count + 1), bytes,
                     alignment, capacity,
                     reinterpret_cast<std::uintptr_t>(token)};
    ++count;
    return true;
  }

  void Seal() noexcept
  {
    baseline = count;
    sealed = true;
  }

  std::uint64_t EventsAfterBaseline() const noexcept
  {
    return sealed && count >= baseline
        ? static_cast<std::uint64_t>(count - baseline)
        : 0;
  }
};

template <typename T>
bool ResizeOwned(std::vector<T>& values,
                 const std::size_t size,
                 const CpuOwnedAllocationSite site,
                 CpuOwnedAllocationLedger& ledger)
{
  const T* before = values.data();
  const std::size_t before_capacity = values.capacity();
  values.resize(size);
  if (values.data() != before || values.capacity() != before_capacity)
  {
    return ledger.Observe(site, values.capacity() * sizeof(T), alignof(T),
                          values.capacity(), values.data());
  }
  return true;
}

bool ValidSlot(const CpuStateSlot& slot) noexcept
{
  return slot.interior.cell && slot.trace.cell &&
      slot.interior.cell == slot.trace.cell && slot.interior.cell->backing;
}

bool CopySlot(const CpuStateSlot& source, CpuStateSlot& destination) noexcept
{
  if (!ValidSlot(source) || !ValidSlot(destination)) { return false; }
  const auto& from = *source.interior.cell->backing;
  auto& to = *destination.interior.cell->backing;
  if (from.interior.size() != to.interior.size() ||
      from.trace.size() != to.trace.size() ||
      to.interior.capacity() < from.interior.size() ||
      to.trace.capacity() < from.trace.size())
  {
    return false;
  }
  std::copy(from.interior.begin(), from.interior.end(), to.interior.begin());
  std::copy(from.trace.begin(), from.trace.end(), to.trace.begin());
  return true;
}

CpuStateSlot MakeSlot(const std::size_t interior_size,
                      const std::size_t trace_size,
                      CpuOwnedAllocationLedger& ledger)
{
  auto cell = std::make_shared<CpuSlotCell>();
  cell->backing = std::make_unique<CpuSlotStorage>();
  if (!ledger.Observe(CpuOwnedAllocationSite::slot_backing,
                      sizeof(CpuSlotStorage), alignof(CpuSlotStorage), 1,
                      cell->backing.get()))
  {
    throw std::bad_alloc();
  }
  if (!ResizeOwned(cell->backing->interior, interior_size,
                   CpuOwnedAllocationSite::slot_interior, ledger) ||
      !ResizeOwned(cell->backing->trace, trace_size,
                   CpuOwnedAllocationSite::slot_trace, ledger))
  {
    throw std::bad_alloc();
  }
  return {{cell}, {cell}};
}

const CpuSlotStorage* Storage(const CpuStateSlot& slot) noexcept
{
  return ValidSlot(slot) ? slot.interior.cell->backing.get() : nullptr;
}

CpuSlotStorage* Storage(CpuStateSlot& slot) noexcept
{
  return ValidSlot(slot) ? slot.interior.cell->backing.get() : nullptr;
}

} // namespace

CpuPersistentConfig MakeCpuSmokeConfig() noexcept
{
  CpuPersistentConfig config;
  config.experiment = StableIdentity::Copy(expected_experiment);
  config.source = StableIdentity::Copy(expected_source);
  return config;
}

CpuPersistentConfig MakeCpuFunctionalConfig(
    const int elements_per_direction) noexcept
{
  CpuPersistentConfig config = MakeCpuSmokeConfig();
  config.scenario = CpuScenario::functional;
  config.polynomial_order = cpu_functional_polynomial_order;
  config.elements_per_direction = elements_per_direction;
  config.assembly_quadrature_order =
      cpu_functional_assembly_quadrature_order;
  return config;
}

bool CpuStateStoragePolicy::PrepareCandidate(
    const CpuStateSlot& committed, CpuStateSlot& candidate) noexcept
{
  return CopySlot(committed, candidate);
}

bool CpuStateStoragePolicy::PrepareRestore(
    const CpuStateSlot& loaded, CpuStateSlot& candidate) noexcept
{
  return CopySlot(loaded, candidate);
}

bool CpuStateStoragePolicy::StoreCandidate(
    CpuStateSlot& candidate, const CpuStateSlot& value) noexcept
{
  return CopySlot(value, candidate);
}

bool CpuStateStoragePolicy::CopyOut(
    const CpuStateSlot& source, CpuStateSlot& copy) noexcept
{
  return CopySlot(source, copy);
}

void CpuStateStoragePolicy::DiscardCandidate(CpuStateSlot& candidate) noexcept
{
  // Candidate authority is invalidated by its attempt-qualified generation.
  // Discard is intentionally byte preserving: poisoning is neither authority
  // revocation nor evidence of it, and it obscures exact rollback audits.
  (void)candidate;
}

bool CpuStateStoragePolicy::CanPublish(
    const CpuStateSlot& committed, const CpuStateSlot& candidate) noexcept
{
  return Disjoint(committed, candidate) &&
      Storage(committed)->interior.size() == Storage(candidate)->interior.size() &&
      Storage(committed)->trace.size() == Storage(candidate)->trace.size();
}

void CpuStateStoragePolicy::Publish(
    CpuStateSlot& committed, CpuStateSlot& candidate) noexcept
{
  // The sole publication mutation: one ownership-handle swap publishes both
  // interior and true-trace coefficient arrays together.
  committed.interior.cell->backing.swap(candidate.interior.cell->backing);
}

void CpuStateStoragePolicy::RevertPublish(
    CpuStateSlot& committed, CpuStateSlot& candidate) noexcept
{
  // The complete-backing exchange is its own exact byte inverse.
  committed.interior.cell->backing.swap(candidate.interior.cell->backing);
}

bool CpuStateStoragePolicy::Disjoint(
    const CpuStateSlot& left, const CpuStateSlot& right) noexcept
{
  return ValidSlot(left) && ValidSlot(right) &&
      left.interior.cell != right.interior.cell &&
      left.interior.cell->backing.get() != right.interior.cell->backing.get() &&
      (left.interior.cell->backing->interior.empty() ||
       right.interior.cell->backing->interior.empty() ||
       left.interior.cell->backing->interior.data() !=
           right.interior.cell->backing->interior.data()) &&
      (left.interior.cell->backing->trace.empty() ||
       right.interior.cell->backing->trace.empty() ||
       left.interior.cell->backing->trace.data() !=
           right.interior.cell->backing->trace.data());
}

StateHash CpuStateStoragePolicy::HashInterior(
    const CpuInteriorState& state) noexcept
{
  if (!state.cell || !state.cell->backing) { return {}; }
  return RankCommonHash(state.cell->backing->interior);
}

StateHash CpuStateStoragePolicy::HashTrace(const CpuTraceState& state) noexcept
{
  if (!state.cell || !state.cell->backing) { return {}; }
  return RankCommonHash(state.cell->backing->trace);
}

CpuCoefficientView CpuCheckpoint::interior() const noexcept
{
  const CpuSlotStorage* storage = Storage(state);
  return storage ? CpuCoefficientView{storage->interior.data(),
                                      storage->interior.size(),
                                      storage->interior.capacity()}
                 : CpuCoefficientView{};
}

CpuCoefficientView CpuCheckpoint::trace() const noexcept
{
  const CpuSlotStorage* storage = Storage(state);
  return storage ? CpuCoefficientView{storage->trace.data(),
                                      storage->trace.size(),
                                      storage->trace.capacity()}
                 : CpuCoefficientView{};
}

bool ImportCpuCheckpoint(const CheckpointDescriptor& descriptor,
                         const double* local_interior,
                         const std::size_t local_interior_size,
                         const double* local_true_trace,
                         const std::size_t local_true_trace_size,
                         CpuCheckpoint& checkpoint) noexcept
{
  const bool local_arguments_ready = !checkpoint.valid &&
      descriptor.successful_commit &&
      (local_interior || local_interior_size == 0) &&
      (local_true_trace || local_true_trace_size == 0) &&
      local_interior_size <=
          std::numeric_limits<std::size_t>::max() / sizeof(double) &&
      local_true_trace_size <=
          std::numeric_limits<std::size_t>::max() / sizeof(double);
  // Phase 1: one fixed admission record gathers local readiness and the full
  // descriptor identity.  Every subsequent return is behind a consensus.
  if (!AdmitCpuCheckpointImport(descriptor, local_arguments_ready))
  {
    return false;
  }

  CpuOwnedAllocationLedger checkpoint_allocations;
  CpuStateSlot imported{};
  bool local_prepared = false;
  try
  {
    imported = MakeSlot(local_interior_size, local_true_trace_size,
                        checkpoint_allocations);
    CpuSlotStorage* storage = Storage(imported);
    if (storage)
    {
      if (local_interior_size != 0)
      {
        std::memcpy(storage->interior.data(), local_interior,
                    local_interior_size * sizeof(double));
      }
      if (local_true_trace_size != 0)
      {
        std::memcpy(storage->trace.data(), local_true_trace,
                    local_true_trace_size * sizeof(double));
      }
      local_prepared = storage->interior.size() == local_interior_size &&
          storage->trace.size() == local_true_trace_size;
    }
  }
  catch (...)
  {
    local_prepared = false;
  }
  // Phase 2: allocation/copy exceptions are reduced after every rank has
  // attempted its local preparation.
  if (!AllRanks(local_prepared)) { return false; }

  const CpuSlotStorage* storage = Storage(imported);
  const bool local_interior_finite = storage && Finite(storage->interior);
  const bool local_trace_finite = storage && Finite(storage->trace);
  // Phase 3: no rank enters a rank-common hash unless all local bytes are
  // finite and both local checks have run.
  if (!AllRanks(local_interior_finite && local_trace_finite)) { return false; }

  // The preparation and finite consensuses make both vectors valid on every
  // rank, so neither rank-common hash can be skipped by a local guard.
  const StateHash imported_interior_hash = RankCommonHash(storage->interior);
  const StateHash imported_trace_hash = RankCommonHash(storage->trace);
  const bool local_hashes_match =
      imported_interior_hash == descriptor.interior_hash &&
      imported_trace_hash == descriptor.trace_hash;
  // Phase 6 follows the ordered interior/trace hash collectives (phases 4/5).
  if (!AllRanks(local_hashes_match)) { return false; }

  checkpoint.descriptor = descriptor;
  checkpoint.state = std::move(imported);
  checkpoint.valid = true;
  return true;
}

CollectiveAdmissionReport CpuWorldCollectivePolicy::Admit(
    const AttemptCollectiveContext& context, const bool prepared,
    const FailureReason reason, const std::uint32_t expected_ranks) noexcept
{
  CollectiveAdmissionReport report;
  report.context = context;
  int ranks = 0;
  report.identity_consistent = ContextsEqual(
      context, [](const auto& left, const auto& right)
      { return ::dg::hdg::runtime::detail::SameContext(left, right); }, ranks);
  report.participating_ranks = static_cast<std::uint32_t>(ranks);
  int local_prepared = prepared ? 1 : 0;
  int all_prepared = 0;
  unsigned long long local_mask = FailureMask(reason);
  unsigned long long global_mask = 0;
  const int prepared_status = MPI_Allreduce(
      &local_prepared, &all_prepared, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  const int failure_status = MPI_Allreduce(
      &local_mask, &global_mask, 1, MPI_UNSIGNED_LONG_LONG, MPI_BOR,
      MPI_COMM_WORLD);
  const bool reductions = prepared_status == MPI_SUCCESS &&
      failure_status == MPI_SUCCESS;
  report.failure_consistent = reductions &&
      report.participating_ranks == expected_ranks;
  report.all_prepared = all_prepared == 1;
  report.failure_mask = static_cast<std::uint64_t>(global_mask);
  report.any_failure = report.failure_mask != 0;
  report.resolved_reason = FirstFailure(report.failure_mask);
  return report;
}

CollectiveGateReport CpuWorldCollectivePolicy::Reduce(
    const AttemptCollectiveContext& context, const GateReport& gates,
    const FailureReason reason, const std::uint32_t expected_ranks) noexcept
{
  CollectiveGateReport report;
  report.context = context;
  int ranks = 0;
  report.identity_consistent = ContextsEqual(
      context, [](const auto& left, const auto& right)
      { return ::dg::hdg::runtime::detail::SameContext(left, right); }, ranks);
  report.participating_ranks = static_cast<std::uint32_t>(ranks);
  const std::array<int, 6> local{{gates.finite ? 1 : 0,
                                  gates.admissible ? 1 : 0,
                                  gates.recovery ? 1 : 0,
                                  gates.true_residual ? 1 : 0,
                                  gates.telemetry ? 1 : 0,
                                  gates.success ? 1 : 0}};
  std::array<int, 6> reduced{};
  unsigned long long local_mask = FailureMask(reason);
  unsigned long long global_mask = 0;
  const int gates_status = MPI_Allreduce(
      local.data(), reduced.data(), static_cast<int>(local.size()), MPI_INT,
      MPI_MIN, MPI_COMM_WORLD);
  const int failure_status = MPI_Allreduce(
      &local_mask, &global_mask, 1, MPI_UNSIGNED_LONG_LONG, MPI_BOR,
      MPI_COMM_WORLD);
  const bool reductions = gates_status == MPI_SUCCESS &&
      failure_status == MPI_SUCCESS;
  report.failure_consistent = reductions &&
      report.participating_ranks == expected_ranks;
  report.gates = {reduced[0] == 1, reduced[1] == 1, reduced[2] == 1,
                  reduced[3] == 1, reduced[4] == 1, reduced[5] == 1};
  report.failure_mask = static_cast<std::uint64_t>(global_mask);
  report.any_failure = report.failure_mask != 0;
  report.resolved_reason = FirstFailure(report.failure_mask);
  return report;
}

CollectiveRestoreReport CpuWorldCollectivePolicy::ReduceRestore(
    const RestoreCollectiveContext& context,
    const RestartCompatibility compatibility,
    const FailureReason reason, const std::uint32_t expected_ranks) noexcept
{
  CollectiveRestoreReport report;
  report.context = context;
  int ranks = 0;
  report.identity_consistent = ContextsEqual(
      context, [](const auto& left, const auto& right)
      { return ::dg::hdg::runtime::detail::SameRestoreContext(left, right); }, ranks);
  report.participating_ranks = static_cast<std::uint32_t>(ranks);
  unsigned long long local_failure = FailureMask(reason);
  unsigned long long global_failure = 0;
  unsigned long long local_compatibility = CompatibilityMask(compatibility);
  unsigned long long global_compatibility = 0;
  const int failure_status = MPI_Allreduce(
      &local_failure, &global_failure, 1, MPI_UNSIGNED_LONG_LONG, MPI_BOR,
      MPI_COMM_WORLD);
  const int compatibility_status = MPI_Allreduce(
      &local_compatibility, &global_compatibility, 1, MPI_UNSIGNED_LONG_LONG,
      MPI_BOR, MPI_COMM_WORLD);
  const bool reductions = failure_status == MPI_SUCCESS &&
      compatibility_status == MPI_SUCCESS;
  report.failure_consistent = reductions &&
      report.participating_ranks == expected_ranks;
  report.failure_mask = static_cast<std::uint64_t>(global_failure);
  report.compatibility_mask =
      static_cast<std::uint64_t>(global_compatibility);
  report.any_failure = report.failure_mask != 0 ||
      report.compatibility_mask != 0;
  report.resolved_reason = report.any_failure
      ? FailureReason::restart_incompatible : FailureReason::none;
  report.resolved_compatibility =
      FirstCompatibility(report.compatibility_mask);
  return report;
}

namespace
{

struct LocalSystem
{
  int elements = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  std::vector<double> residual_interior;
  std::vector<double> residual_trace;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> d;
  std::vector<double> f;
  std::vector<double> g;

  bool Allocate(const int batch_count, const int ni, const int nt,
                CpuOwnedAllocationLedger& ledger)
  {
    elements = batch_count;
    interior_dofs = ni;
    trace_dofs = nt;
    const std::size_t batches = static_cast<std::size_t>(elements);
    const std::size_t interior = static_cast<std::size_t>(ni);
    const std::size_t trace = static_cast<std::size_t>(nt);
    return ResizeOwned(residual_interior, batches * interior,
                       CpuOwnedAllocationSite::local_system_vector, ledger) &&
        ResizeOwned(residual_trace, batches * trace,
                    CpuOwnedAllocationSite::local_system_vector, ledger) &&
        ResizeOwned(a, batches * interior * interior,
                    CpuOwnedAllocationSite::local_system_vector, ledger) &&
        ResizeOwned(b, batches * interior * trace,
                    CpuOwnedAllocationSite::local_system_vector, ledger) &&
        ResizeOwned(c, batches * trace * interior,
                    CpuOwnedAllocationSite::local_system_vector, ledger) &&
        ResizeOwned(d, batches * trace * trace,
                    CpuOwnedAllocationSite::local_system_vector, ledger) &&
        ResizeOwned(f, batches * interior,
                    CpuOwnedAllocationSite::local_system_vector, ledger) &&
        ResizeOwned(g, batches * trace,
                    CpuOwnedAllocationSite::local_system_vector, ledger);
  }

  void Zero() noexcept
  {
    std::fill(residual_interior.begin(), residual_interior.end(), 0.0);
    std::fill(residual_trace.begin(), residual_trace.end(), 0.0);
    std::fill(a.begin(), a.end(), 0.0);
    std::fill(b.begin(), b.end(), 0.0);
    std::fill(c.begin(), c.end(), 0.0);
    std::fill(d.begin(), d.end(), 0.0);
    std::fill(f.begin(), f.end(), 0.0);
    std::fill(g.begin(), g.end(), 0.0);
  }
};

void CopySystem(const LocalSystem& source, LocalSystem& destination) noexcept
{
  std::copy(source.residual_interior.begin(), source.residual_interior.end(),
            destination.residual_interior.begin());
  std::copy(source.residual_trace.begin(), source.residual_trace.end(),
            destination.residual_trace.begin());
  std::copy(source.a.begin(), source.a.end(), destination.a.begin());
  std::copy(source.b.begin(), source.b.end(), destination.b.begin());
  std::copy(source.c.begin(), source.c.end(), destination.c.begin());
  std::copy(source.d.begin(), source.d.end(), destination.d.begin());
  std::copy(source.f.begin(), source.f.end(), destination.f.begin());
  std::copy(source.g.begin(), source.g.end(), destination.g.begin());
}

bool Assign(mfem::Vector& destination,
            const std::vector<double>& source) noexcept
{
  if (destination.Size() != static_cast<int>(source.size())) { return false; }
  double* values = destination.HostWrite();
  if (values == nullptr && !source.empty()) { return false; }
  if (!source.empty())
  {
    std::memcpy(values, source.data(), source.size() * sizeof(double));
  }
  return true;
}

bool Copy(const mfem::Vector& source,
          std::vector<double>& destination) noexcept
{
  if (source.Size() != static_cast<int>(destination.size())) { return false; }
  const double* values = source.HostRead();
  if (values == nullptr && !destination.empty()) { return false; }
  if (!destination.empty())
  {
    std::memcpy(destination.data(), values,
                destination.size() * sizeof(double));
  }
  return Finite(destination);
}

StateHash HashSystemMatrix(const LocalSystem& system) noexcept
{
  StateHash value = HashDoubles(system.a);
  value = HashPair(value, HashDoubles(system.b));
  value = HashPair(value, HashDoubles(system.c));
  return HashPair(value, HashDoubles(system.d));
}

template <typename Case>
bool ExactW8cConfig(const CpuPersistentConfig& config) noexcept
{
  return config.experiment.valid() &&
      config.experiment == StableIdentity::Copy(Case::experiment_id) &&
      config.source.valid() &&
      config.source == StableIdentity::Copy(Case::conventional_source_id) &&
      config.scenario == CpuScenario::smoke &&
      config.polynomial_order == Case::polynomial_order &&
      config.elements_per_direction == static_cast<int>(Case::mesh_n) &&
      config.assembly_quadrature_order == Case::conventional_q_a &&
      ::dg::hdg::runtime::detail::DoubleBits(config.dt) ==
          ::dg::hdg::runtime::detail::DoubleBits(Case::dt);
}

bool ExactConfig(const CpuPersistentConfig& config) noexcept
{
  const bool w8c_config = ExactW8cConfig<w8c::ExactJumpV1Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionV2Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionN16Dt11Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionN16Dt12Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionN32Dt12Case>(config);
  const bool smoke = config.scenario == CpuScenario::smoke &&
      config.polynomial_order == cpu_smoke_polynomial_order &&
      config.elements_per_direction == cpu_smoke_elements_per_direction &&
      config.assembly_quadrature_order ==
          cpu_smoke_assembly_quadrature_order;
  const bool functional_shape =
      config.elements_per_direction ==
          cpu_functional_minimum_elements_per_direction ||
      config.elements_per_direction ==
          cpu_functional_principal_elements_per_direction ||
      config.elements_per_direction ==
          cpu_functional_maximum_elements_per_direction;
  const bool functional = config.scenario == CpuScenario::functional &&
      config.polynomial_order == cpu_functional_polynomial_order &&
      functional_shape && config.assembly_quadrature_order ==
          cpu_functional_assembly_quadrature_order;
  const bool w8a_config = config.experiment.valid() && config.experiment ==
          StableIdentity::Copy(expected_experiment) &&
      config.source.valid() && config.source ==
          StableIdentity::Copy(expected_source) &&
      (smoke || functional) &&
      ::dg::hdg::runtime::detail::DoubleBits(config.dt) ==
          ::dg::hdg::runtime::detail::DoubleBits(cpu_backward_euler_dt) &&
      config.maximum_newton_updates == cpu_maximum_newton_updates &&
      config.maximum_line_search_trials == cpu_maximum_line_search_trials &&
      ::dg::hdg::runtime::detail::DoubleBits(config.armijo_constant) ==
          ::dg::hdg::runtime::detail::DoubleBits(cpu_armijo_constant) &&
      ::dg::hdg::runtime::detail::DoubleBits(config.nonlinear_tolerance) ==
          ::dg::hdg::runtime::detail::DoubleBits(cpu_nonlinear_tolerance) &&
      ::dg::hdg::runtime::detail::DoubleBits(config.true_residual_tolerance) ==
          ::dg::hdg::runtime::detail::DoubleBits(cpu_nonlinear_tolerance) &&
      config.maximum_linear_iterations == cpu_maximum_linear_iterations &&
      ::dg::hdg::runtime::detail::DoubleBits(
          config.linear_relative_tolerance) ==
          ::dg::hdg::runtime::detail::DoubleBits(
              cpu_linear_relative_tolerance) &&
      ::dg::hdg::runtime::detail::DoubleBits(
          config.conservation_relative_tolerance) ==
          ::dg::hdg::runtime::detail::DoubleBits(
              cpu_conservation_relative_tolerance) &&
      config.allow_one_retry;
  const bool common_controls =
      config.maximum_newton_updates == cpu_maximum_newton_updates &&
      config.maximum_line_search_trials == cpu_maximum_line_search_trials &&
      ::dg::hdg::runtime::detail::DoubleBits(config.armijo_constant) ==
          ::dg::hdg::runtime::detail::DoubleBits(cpu_armijo_constant) &&
      ::dg::hdg::runtime::detail::DoubleBits(config.nonlinear_tolerance) ==
          ::dg::hdg::runtime::detail::DoubleBits(cpu_nonlinear_tolerance) &&
      ::dg::hdg::runtime::detail::DoubleBits(
          config.true_residual_tolerance) ==
          ::dg::hdg::runtime::detail::DoubleBits(cpu_nonlinear_tolerance) &&
      config.maximum_linear_iterations == cpu_maximum_linear_iterations &&
      ::dg::hdg::runtime::detail::DoubleBits(
          config.linear_relative_tolerance) ==
          ::dg::hdg::runtime::detail::DoubleBits(
              cpu_linear_relative_tolerance) &&
      ::dg::hdg::runtime::detail::DoubleBits(
          config.conservation_relative_tolerance) ==
          ::dg::hdg::runtime::detail::DoubleBits(
              cpu_conservation_relative_tolerance) &&
      config.allow_one_retry &&
      !config.force_private_scientific_gate_failure;
  return w8a_config || (w8c_config && common_controls);
}

bool IsW8cConfig(const CpuPersistentConfig& config) noexcept
{
  return ExactW8cConfig<w8c::ExactJumpV1Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionV2Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionN16Dt11Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionN16Dt12Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionN32Dt12Case>(config);
}

bool IsSmoothTransitionConfig(const CpuPersistentConfig& config) noexcept
{
  return ExactW8cConfig<w8c::SmoothTransitionV2Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionN16Dt11Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionN16Dt12Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionN32Dt12Case>(config);
}

std::uint64_t W8cAcceptedSteps(const CpuPersistentConfig& config) noexcept
{
  if (ExactW8cConfig<w8c::SmoothTransitionN16Dt11Case>(config))
    return w8c::SmoothTransitionN16Dt11Case::accepted_steps;
  if (ExactW8cConfig<w8c::SmoothTransitionN16Dt12Case>(config) ||
      ExactW8cConfig<w8c::SmoothTransitionN32Dt12Case>(config))
    return w8c::SmoothTransitionN16Dt12Case::accepted_steps;
  return w8c::ExactJumpV1Case::accepted_steps;
}

bool ExactRankShape(const CpuPersistentConfig& config,
                    const int ranks) noexcept
{
  return ranks == 1 ||
      (ranks == 2 &&
       (config.scenario == CpuScenario::smoke ||
        config.elements_per_direction ==
            cpu_functional_principal_elements_per_direction));
}

bool ExactConfigConsensus(const CpuPersistentConfig& config) noexcept
{
  const std::array<std::uint64_t, 15> local{{
      static_cast<std::uint64_t>(config.scenario),
      static_cast<std::uint64_t>(config.polynomial_order),
      static_cast<std::uint64_t>(config.elements_per_direction),
      static_cast<std::uint64_t>(config.assembly_quadrature_order),
      static_cast<std::uint64_t>(config.maximum_newton_updates),
      static_cast<std::uint64_t>(config.maximum_linear_iterations),
      static_cast<std::uint64_t>(config.maximum_line_search_trials),
      ::dg::hdg::runtime::detail::DoubleBits(config.dt),
      ::dg::hdg::runtime::detail::DoubleBits(config.armijo_constant),
      ::dg::hdg::runtime::detail::DoubleBits(config.nonlinear_tolerance),
      ::dg::hdg::runtime::detail::DoubleBits(config.linear_relative_tolerance),
      ::dg::hdg::runtime::detail::DoubleBits(config.true_residual_tolerance),
      ::dg::hdg::runtime::detail::DoubleBits(
          config.conservation_relative_tolerance),
      static_cast<std::uint64_t>(config.allow_one_retry),
      static_cast<std::uint64_t>(
          config.force_private_scientific_gate_failure)}};
  std::array<std::uint64_t, 15> minimum{}, maximum{};
  return MPI_Allreduce(local.data(), minimum.data(),
                       static_cast<int>(local.size()), MPI_UINT64_T, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS &&
      MPI_Allreduce(local.data(), maximum.data(),
                    static_cast<int>(local.size()), MPI_UINT64_T, MPI_MAX,
                    MPI_COMM_WORLD) == MPI_SUCCESS &&
      minimum == maximum;
}

bool ExactTuple(const RuntimeTupleDescriptor& tuple,
                const int ranks) noexcept
{
  const StableIdentity expected_backend = StableIdentity::Copy(
      ranks == 1 ? std::string_view{"cpu.serial"}
                 : std::string_view{"cpu.rank2"});
  return tuple.Valid() && tuple.model == StableIdentity::Copy(expected_model) &&
      tuple.formulation == StableIdentity::Copy(expected_formulation) &&
      tuple.trace == StableIdentity::Copy(expected_trace) &&
      tuple.stabilization == StableIdentity::Copy(expected_stabilization) &&
      tuple.time_treatment == StableIdentity::Copy(expected_time) &&
      tuple.backend == expected_backend &&
      tuple.precision == StableIdentity::Copy(expected_precision);
}

StateHash HashTopologyIntegers(const std::vector<int>& values) noexcept
{
  return HashBytes(values.data(), values.size() * sizeof(int));
}

int PeriodicIndex(const double coordinate,
                  const int elements_per_direction) noexcept
{
  int value = static_cast<int>(std::llround(
      coordinate * elements_per_direction));
  value %= elements_per_direction;
  if (value < 0) { value += elements_per_direction; }
  return value;
}

int CellIndex(const double coordinate,
              const int elements_per_direction) noexcept
{
  double periodic = coordinate - std::floor(coordinate);
  int value = static_cast<int>(std::floor(
      periodic * elements_per_direction + 64.0 *
          std::numeric_limits<double>::epsilon()));
  return std::max(0, std::min(elements_per_direction - 1, value));
}

} // namespace

struct CpuNewtonWorkspace
{
  StateHash candidate_state_hash{};
  std::uint64_t candidate_generation = 0;
  bool accepted = false;
  bool ordered = false;
  bool affine_consumption_verified = false;
  bool retained_for_publication = false;
};

struct CpuAttemptWorkspace
{
  CpuAttemptEvidence attempt{};
  CpuNewtonWorkspace current_newton{};
  double maximum_conservation_error = 0.0;
  double maximum_conservation_relative_error = 0.0;
  bool conservation_gate = false;
};

struct detail::CpuNumericsCore::Implementation
{
  enum class ArtifactStage : unsigned char
  {
    local,
    condensed,
    solve,
    solution,
    recovery
  };
  struct ArtifactCapability
  {
    ArtifactCapability() noexcept = default;
    ArtifactCapability(const std::uint64_t sequence,
                       const std::uint64_t cursor,
                       const std::uint64_t artifact_generation,
                       const ArtifactStage artifact_stage) noexcept
        : attempt_sequence(sequence), candidate_cursor(cursor),
          generation(artifact_generation), stage(artifact_stage), valid(true)
    {}
    ArtifactCapability(const ArtifactCapability&) = delete;
    ArtifactCapability& operator=(const ArtifactCapability&) = delete;
    ArtifactCapability(ArtifactCapability&& other) noexcept
    {
      MoveFrom(other);
    }
    ArtifactCapability& operator=(ArtifactCapability&& other) noexcept
    {
      if (this != &other) { MoveFrom(other); }
      return *this;
    }
    std::uint64_t attempt_sequence = 0;
    std::uint64_t candidate_cursor = 0;
    std::uint64_t generation = 0;
    ArtifactStage stage = ArtifactStage::local;
    bool valid = false;
  private:
    void MoveFrom(ArtifactCapability& other) noexcept
    {
      attempt_sequence = other.attempt_sequence;
      candidate_cursor = other.candidate_cursor;
      generation = other.generation;
      stage = other.stage;
      valid = other.valid;
      other.valid = false;
    }
  };
  struct SparseSchurTerm
  {
    std::size_t schur_index = 0;
    int data_index = -1;
    double sign = 1.0;
    bool offdiagonal = false;
  };
  struct SparseOutgoingTerm
  {
    std::size_t schur_index = 0;
    double sign = 1.0;
    int destination_rank = -1;
  };
  struct SparseIncomingSlot
  {
    int data_index = -1;
    bool offdiagonal = false;
  };

  CpuPersistentConfig config{};
  RuntimeTupleDescriptor tuple{};
  const generic::EulerHdgRequest* borrowed_request = nullptr;
  MPI_Comm communicator = MPI_COMM_WORLD;
  std::uint64_t incarnation = 0;
  int rank = -1;
  int ranks = 0;
  bool initialized = false;
  CpuOwnedAllocationLedger allocations{};

  // The established standalone adapter owns the mesh it constructs.  The
  // generic entry borrows the caller mesh for the duration of RunBorrowedEuler
  // and never places it in an owning handle.
  std::unique_ptr<mfem::ParMesh> owned_mesh;
  mfem::ParMesh* mesh = nullptr;
  std::unique_ptr<mfem::L2_FECollection> interior_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> interior_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
  std::optional<trace::DGInterfaceTraceExchangePlan> exchange_plan;

  CpuStateSlot committed_slot{};
  CpuStateSlot candidate_slot{};
  std::vector<double> first_iterate_interior;
  std::vector<double> first_iterate_trace;

  // IGR augments the conventional spaces and lifecycle in place.  The first
  // four component slabs remain the public Euler contract; the trailing
  // sigma/z and sigma-hat slabs are private state in those same slots.
  bool igr_active = false;
  int interior_components = cpu_conserved_components;
  int trace_components = cpu_conserved_components;

  std::unique_ptr<mfem::ParGridFunction> old_interior;
  std::unique_ptr<mfem::ParGridFunction> external_source;
  std::unique_ptr<mfem::ParGridFunction> iterate_interior;
  std::unique_ptr<mfem::ParGridFunction> trial_interior;
  std::unique_ptr<mfem::ParGridFunction> iterate_trace;
  std::unique_ptr<mfem::ParGridFunction> trial_trace;
  mfem::Vector iterate_true_trace;
  mfem::Vector trial_true_trace;
  mfem::Vector true_increment;
  mfem::Vector trace_distribution_workspace;
  mfem::Vector distributed_increment;
  mfem::Vector local_trace_rhs;
  mfem::Vector true_trace_rhs;
  mfem::Vector local_trace_residual;
  mfem::Vector true_trace_residual;
  mfem::Vector volume_shape_workspace;
  mfem::DenseMatrix volume_derivative_workspace;
  mfem::Vector physical_workspace;
  mfem::Vector face_interior_shape_workspace;
  mfem::Vector face_trace_shape_workspace;
  mfem::Vector scaled_normal_workspace;
  mfem::Array<int> element_trace_dofs_workspace;
  mfem::DenseMatrix local_schur_workspace;

  int local_interior = 0;
  int local_trace = 0;
  int scalar_interior_dofs = 0;
  int scalar_trace_dofs = 0;
  std::vector<int> element_interior_vdofs;
  std::vector<int> element_faces;
  std::vector<unsigned char> shared_faces;
  std::vector<int> signed_trace_dofs;
  // Extracted only while the serial attribute tag is present.  These are
  // provenance for the MFEM-selected local mesh, never physics attributes.
  std::vector<std::uint64_t> local_parent_cell_ids;
  std::vector<std::uint64_t> local_parent_cell_ids_sorted;
  std::vector<std::uint64_t> interior_global_ids;
  std::vector<std::uint64_t> physical_interior_ids;
  std::vector<std::uint64_t> trace_canonical_ids;

  LocalSystem current_system;
  LocalSystem trial_system;
  core::HostReferenceEngine condensation;
  std::vector<double> schur;
  std::vector<double> reduced_rhs;
  std::vector<double> local_trace_increment;
  std::vector<double> recovered_interior;
  std::vector<double> grid_interior_increment;
  std::vector<int> interior_census;

  std::unique_ptr<mfem::SparseMatrix> local_matrix;
  std::unique_ptr<mfem::ParBilinearForm> parallel_form;
  std::unique_ptr<mfem::HypreParMatrix> true_matrix;
  mfem::SparseMatrix true_diagonal_view;
  mfem::SparseMatrix true_offdiagonal_view;
  HYPRE_BigInt* true_offdiagonal_columns = nullptr;
  std::vector<SparseSchurTerm> owned_schur_terms;
  std::vector<SparseOutgoingTerm> outgoing_schur_terms;
  std::vector<SparseIncomingSlot> incoming_schur_slots;
  std::vector<double> outgoing_schur_values;
  std::vector<double> incoming_schur_values;
  std::vector<int> outgoing_schur_counts;
  std::vector<int> incoming_schur_counts;
  std::vector<int> outgoing_schur_offsets;
  std::vector<int> incoming_schur_offsets;
  std::unique_ptr<mfem::HypreParVector> hypre_rhs;
  std::unique_ptr<mfem::HypreParVector> hypre_solution;
  std::unique_ptr<mfem::HypreParVector> hypre_residual;
  std::unique_ptr<mfem::HypreILU> ilu;
  std::unique_ptr<mfem::HypreGMRES> gmres;

  CpuRunEvidence evidence{};
  CpuAttemptWorkspace active_evidence{};
  StateCoordinates entry_coordinates{};
  StateHash entry_interior_hash{};
  StateHash entry_trace_hash{};
  std::uint64_t next_artifact_generation = 1;
  std::uint64_t next_candidate_generation = 1;
  std::uint64_t next_matrix_refresh_generation = 1;
  std::uint64_t pending_factor_setup_generation = 0;
  std::uint64_t complete_factor_setup_generation = 0;
  std::uint64_t active_candidate_generation = 0;
  StateHash active_candidate_state_hash{};
  bool publication_chain_authority = false;
  bool borrowed_prepare_active = false;
  // Private qualification control.  It is armed only through friend test
  // access and consumed by exactly one borrowed attempt.
  bool exact_zero_initial_residual_for_tests = false;
  double borrowed_prepared_previous_dt = 0.0;
  bool CollectiveAll(const bool local) const noexcept
  {
    int value = local ? 1 : 0;
    int reduced = 0;
    return MPI_Allreduce(&value, &reduced, 1, MPI_INT, MPI_MIN,
                         communicator) == MPI_SUCCESS && reduced == 1;
  }

  bool IsBorrowed() const noexcept { return borrowed_request != nullptr; }

  bool IsIgr() const noexcept { return IsBorrowed() && igr_active; }

  bool FillBorrowedField(mfem::ParGridFunction& field,
                         const generic::EulerFieldView view) const noexcept
  {
    if (view.scalar_count == 0)
    {
      field = 0.0;
      return true;
    }
    double* values = field.HostWrite();
    const mfem::FiniteElementSpace* const space = field.FESpace();
    if ((!values && field.Size() != 0) || !space ||
        space->GetVDim() != interior_components ||
        field.Size() % interior_components != 0)
    {
      return false;
    }
    const int scalar_size = field.Size() / interior_components;
    if (view.components != cpu_conserved_components ||
        view.scalar_count != static_cast<std::size_t>(
            cpu_conserved_components * scalar_size) ||
        (view.scalar_count != 0 && view.data == nullptr))
    {
      return false;
    }
    field = 0.0;
    for (int component = 0; component < cpu_conserved_components; ++component)
      for (int scalar = 0; scalar < scalar_size; ++scalar)
    {
      const int index = component * scalar_size + scalar;
      const double value = view.data[static_cast<std::size_t>(index)];
      if (!Finite(value)) { return false; }
      values[index] = value;
    }
    return true;
  }

  bool FillBorrowedTrueTrace(mfem::Vector& trace,
                             const generic::EulerFieldView view) const noexcept
  {
    // A globally supplied true trace may have an exactly empty local piece on
    // a rank which owns no true dofs.  That rank contributes no bytes but must
    // still take the supplied-trace collective path with its peers.
    if (trace_components <= 0 || trace.Size() % trace_components != 0)
    {
      return false;
    }
    const int scalar_size = trace.Size() / trace_components;
    const int public_size = cpu_conserved_components * scalar_size;
    if (view.components != cpu_conserved_components ||
        view.scalar_count != static_cast<std::size_t>(public_size) ||
        (public_size != 0 && view.data == nullptr))
    {
      return false;
    }
    trace = 0.0;
    for (int index = 0; index < public_size; ++index)
    {
      const double value = view.data[static_cast<std::size_t>(index)];
      if (!Finite(value)) { return false; }
      trace[index] = value;
    }
    return true;
  }

  bool ConsumeArtifact(ArtifactCapability&& capability,
                       const ArtifactStage expected_stage,
                       const AttemptIdentity& attempt,
                       const std::uint64_t expected_generation) noexcept
  {
    const bool accepted = capability.valid &&
        capability.stage == expected_stage &&
        capability.attempt_sequence == attempt.sequence &&
        capability.candidate_cursor == attempt.sequence &&
        capability.generation == expected_generation;
    capability.valid = false;
    return accepted;
  }

  template <typename T>
  bool ObserveWrapper(const std::unique_ptr<T>& value) noexcept
  {
    return value && allocations.Observe(CpuOwnedAllocationSite::mfem_wrapper,
                                         sizeof(T), alignof(T), 1,
                                         value.get());
  }

  void ApplyFrozenIluSettings() noexcept
  {
    ilu->SetType(0);
    ilu->SetLevelOfFill(0);
    ilu->SetMaxIter(1);
    ilu->SetTol(0.0);
    ilu->SetLocalReordering(0);
    ilu->SetPrintLevel(0);
  }

  bool RebindTraceSolverAfterMatrixRefresh() noexcept
  {
    pending_factor_setup_generation = 0;
    if (!true_matrix || !gmres || !ilu) { return false; }
    try
    {
      // SetOperator invalidates GMRES setup.  For an already-bound HypreILU,
      // pinned MFEM recreates the HYPRE object with defaults here, so the
      // frozen type/fill/iteration/tolerance/reordering/print settings must be
      // reapplied before its callbacks are rebound and Mult performs setup.
      gmres->SetOperator(*true_matrix);
      ApplyFrozenIluSettings();
      gmres->SetPreconditioner(*ilu);
      return true;
    }
    catch (...)
    {
      return false;
    }
  }

  CpuSlotStorage* Committed() noexcept { return Storage(committed_slot); }
  const CpuSlotStorage* Committed() const noexcept
  {
    return Storage(committed_slot);
  }
  CpuSlotStorage* Candidate() noexcept { return Storage(candidate_slot); }
  const CpuSlotStorage* Candidate() const noexcept
  {
    return Storage(candidate_slot);
  }

  bool InjectionArguments(const CpuFailureInjection& injection,
                          const std::uint64_t target_step) const noexcept
  {
    const int local[4] = {static_cast<int>(injection.phase),
                          static_cast<int>(injection.product_fault),
                          injection.source_rank,
                          static_cast<int>(injection.target_step)};
    std::array<int, 4> minimum{}, maximum{};
    const int minimum_status = MPI_Allreduce(
        local, minimum.data(), 4, MPI_INT, MPI_MIN, communicator);
    const int maximum_status = MPI_Allreduce(
        local, maximum.data(), 4, MPI_INT, MPI_MAX, communicator);
    const bool reduced = minimum_status == MPI_SUCCESS &&
        maximum_status == MPI_SUCCESS;
    const bool identical = minimum == maximum;
    const bool phase_in_range =
        static_cast<int>(injection.phase) >=
            static_cast<int>(CpuFailurePhase::none) &&
        static_cast<int>(injection.phase) <=
            static_cast<int>(CpuFailurePhase::terminal_evidence);
    const bool product_in_range =
        static_cast<int>(injection.product_fault) >=
            static_cast<int>(CpuProductFault::none) &&
        static_cast<int>(injection.product_fault) <=
            static_cast<int>(CpuProductFault::out_of_order_matrix);
    const bool off = injection.phase == CpuFailurePhase::none;
    const bool source = off ? injection.source_rank == -1
                            : injection.source_rank >= 0 &&
                                  injection.source_rank < ranks;
    const bool target = off ? injection.target_step == 0
                            : injection.target_step == target_step;
    const bool owned_phase = off ||
        injection.phase == CpuFailurePhase::assembly ||
        injection.phase == CpuFailurePhase::condensation ||
        injection.phase == CpuFailurePhase::trace_solve ||
        injection.phase == CpuFailurePhase::recovery ||
        injection.phase == CpuFailurePhase::after_recovery_before_commit ||
        injection.phase == CpuFailurePhase::diagnostics;
    return reduced && identical && phase_in_range && product_in_range &&
        source && target && owned_phase;
  }

  bool EnterPhase(const CpuFailureInjection& injection,
                  const CpuFailurePhase phase) noexcept
  {
    ++active_evidence.attempt.work.phase_entries;
    const bool local_injected = injection.phase == phase &&
        injection.source_rank == rank;
    const bool admitted = CollectiveAll(!local_injected);
    if (!admitted)
    {
      active_evidence.attempt.failure_phase = phase;
      active_evidence.attempt.failure_source_rank = injection.source_rank;
    }
    return admitted;
  }

  void CompletePhase() noexcept
  {
    ++active_evidence.attempt.work.phase_completions;
  }

  bool ProjectTrace(mfem::ParGridFunction& local_trace,
                    mfem::Vector& true_trace,
                    const bool perturbed) noexcept
  {
    bool local_projection_ready = true;
    try
    {
      if (IsBorrowed())
      {
        local_trace = 0.0;
        for (int face = 0;
             face < mesh->GetNumFaces() && local_projection_ready; ++face)
        {
          mfem::FaceElementTransformations* face_transformation =
              mesh->GetFaceElementTransformations(face);
          const mfem::FiniteElement* trace_element =
              trace_space->GetFaceElement(face);
          mfem::Array<int> vdofs;
          trace_space->GetFaceVDofs(face, vdofs);
          if (!face_transformation || !trace_element ||
              face_transformation->Elem1No < 0 ||
              face_transformation->Elem1No >= mesh->GetNE() ||
              vdofs.Size() != trace_element->GetDof() *
                  trace_components)
          {
            local_projection_ready = false;
            break;
          }
          const mfem::FiniteElement* interior_element = interior_space->GetFE(
              face_transformation->Elem1No);
          const int* const interior_vdofs = element_interior_vdofs.data() +
              static_cast<std::size_t>(face_transformation->Elem1No) *
                  local_interior;
          if (!interior_element ||
              interior_element->GetDof() != scalar_interior_dofs)
          {
            local_projection_ready = false;
            break;
          }
          const mfem::IntegrationRule& nodes = trace_element->GetNodes();
          for (int scalar = 0;
               scalar < trace_element->GetDof() && local_projection_ready;
               ++scalar)
          {
            const mfem::IntegrationPoint& face_point = nodes.IntPoint(scalar);
            face_transformation->SetAllIntPoints(&face_point);
            interior_element->CalcShape(
                face_transformation->GetElement1IntPoint(),
                face_interior_shape_workspace);
            State state{};
            if (!Interpolate(*old_interior, interior_vdofs,
                             scalar_interior_dofs,
                             face_interior_shape_workspace, state))
            {
              local_projection_ready = false;
              break;
            }
            for (int component = 0; component < cpu_conserved_components;
                 ++component)
            {
              double sign = 1.0;
              const int dof = Decode(vdofs[ByNodes(
                  scalar, trace_element->GetDof(), component)], sign);
              if (dof < 0 || dof >= local_trace.Size())
              {
                local_projection_ready = false;
                break;
              }
              local_trace[dof] = sign * state[component];
            }
          }
        }
      }
      else
      {
      local_trace = 0.0;
      for (int face = 0;
           face < mesh->GetNumFaces() && local_projection_ready; ++face)
      {
        const mfem::FiniteElement* finite_element =
            trace_space->GetFaceElement(face);
        mfem::ElementTransformation* transformation =
            mesh->GetFaceTransformation(face);
        if (!finite_element || !transformation)
        {
          local_projection_ready = false;
          break;
        }
        mfem::Array<int> vdofs;
        trace_space->GetFaceVDofs(face, vdofs);
        const int scalar_dofs = finite_element->GetDof();
        if (vdofs.Size() != scalar_dofs * trace_components)
        {
          local_projection_ready = false;
          break;
        }
        if (IsSmoothTransitionConfig(config))
        {
          std::vector<double> mass(
              static_cast<std::size_t>(scalar_dofs) * scalar_dofs, 0.0);
          std::array<std::vector<double>, cpu_conserved_components> rhs;
          for (auto& values : rhs)
            values.assign(static_cast<std::size_t>(scalar_dofs), 0.0);
          mfem::Vector shape(scalar_dofs), physical(cpu_dimension);
          const mfem::IntegrationRule& rule = mfem::IntRules.Get(
              finite_element->GetGeomType(),
              w8c::SmoothTransitionV2Case::projection_q);
          for (int q = 0; q < rule.GetNPoints(); ++q)
          {
            const mfem::IntegrationPoint& ip = rule.IntPoint(q);
            transformation->SetIntPoint(&ip);
            finite_element->CalcShape(ip, shape);
            transformation->Transform(ip, physical);
            State value{};
            if (!SmoothTransitionConservative(physical[0], value))
            {
              local_projection_ready = false;
              break;
            }
            const double weight = ip.weight * transformation->Weight();
            for (int row = 0; row < scalar_dofs; ++row)
            {
              for (int column = 0; column < scalar_dofs; ++column)
              {
                mass[static_cast<std::size_t>(row) * scalar_dofs + column] +=
                    weight * shape[row] * shape[column];
              }
              for (int component = 0;
                   component < cpu_conserved_components; ++component)
              {
                rhs[static_cast<std::size_t>(component)]
                   [static_cast<std::size_t>(row)] +=
                    weight * shape[row] * value[component];
              }
            }
          }
          ProjectionDenseLu factor;
          local_projection_ready = local_projection_ready &&
              factor.Factor(mass, scalar_dofs);
          for (int component = 0;
               component < cpu_conserved_components && local_projection_ready;
               ++component)
          {
            std::vector<double> coefficients;
            local_projection_ready = factor.Solve(
                rhs[static_cast<std::size_t>(component)], coefficients);
            for (int scalar = 0;
                 scalar < scalar_dofs && local_projection_ready; ++scalar)
            {
              double sign = 1.0;
              const int dof = Decode(
                  vdofs[ByNodes(scalar, scalar_dofs, component)], sign);
              if (dof < 0 || dof >= local_trace.Size())
              {
                local_projection_ready = false;
                break;
              }
              local_trace[dof] = sign *
                  coefficients[static_cast<std::size_t>(scalar)];
            }
          }
          continue;
        }
        const mfem::IntegrationRule& nodes = finite_element->GetNodes();
        for (int scalar = 0;
             scalar < scalar_dofs && local_projection_ready; ++scalar)
        {
          mfem::Vector physical(cpu_dimension);
          transformation->Transform(nodes.IntPoint(scalar), physical);
          State state{};
          const bool ok = IsW8cConfig(config)
              ? W8cTraceConservative(physical[0], state)
              : perturbed
              ? PerturbedConservative(physical[0], physical[1], state)
              : ManufacturedConservative(physical[0], physical[1], state);
          if (!ok)
          {
            local_projection_ready = false;
            break;
          }
          for (int component = 0;
               component < cpu_conserved_components && local_projection_ready;
               ++component)
          {
            double sign = 1.0;
            const int dof = Decode(
                vdofs[ByNodes(scalar, scalar_dofs, component)], sign);
            if (dof < 0 || dof >= local_trace.Size())
            {
              local_projection_ready = false;
              break;
            }
            local_trace[dof] = sign * state[component];
          }
        }
      }
      }
    }
    catch (...)
    {
      local_projection_ready = false;
    }
    if (!CollectiveAll(local_projection_ready)) { return false; }

    bool true_projection_ready = true;
    try
    {
      local_trace.ParallelProject(true_trace);
      true_projection_ready = Finite(true_trace);
    }
    catch (...)
    {
      true_projection_ready = false;
    }
    return CollectiveAll(true_projection_ready);
  }

  bool LoadTrace(const mfem::Vector& true_values,
                 mfem::ParGridFunction& local_values) noexcept
  {
    bool local_prepared = true;
    try
    {
      trace_distribution_workspace = 0.0;
      local_prepared = exchange_plan.has_value() &&
          !ConsumeOwnedCollectiveTestFault(
              detail::CpuCollectiveTestFault::trace_load_preparation, rank,
              incarnation);
    }
    catch (...)
    {
      local_prepared = false;
    }
    if (!CollectiveAll(local_prepared)) { return false; }

    bool distributed = false;
    try
    {
      distributed = exchange_plan->Distribute(
          *trace_space, true_values, trace_distribution_workspace);
    }
    catch (...)
    {
      distributed = false;
    }
    if (!CollectiveAll(distributed)) { return false; }

    bool local_copied = true;
    try
    {
      const double* source = trace_distribution_workspace.HostRead();
      double* destination = local_values.HostWrite();
      if ((!source || !destination) &&
          trace_distribution_workspace.Size() != 0)
      {
        local_copied = false;
      }
      if (local_copied && trace_distribution_workspace.Size() != 0)
      {
        std::memcpy(destination, source,
                    static_cast<std::size_t>(
                        trace_distribution_workspace.Size()) *
                        sizeof(double));
      }
    }
    catch (...)
    {
      local_copied = false;
    }
    return CollectiveAll(local_copied);
  }

  bool BuildCanonicalIds() noexcept
  {
    try
    {
      if (!exchange_plan) { return false; }
      if (IsBorrowed())
      {
        // Generic callers do not promise the standalone periodic provenance
        // labels.  The solver needs the trace exchange map, not those labels.
        // Leave canonical-export metadata empty rather than infer or mutate
        // mesh attributes owned by the caller.
        return true;
      }
      if (!ResizeOwned(interior_global_ids,
                       static_cast<std::size_t>(interior_space->GetVSize()),
                       CpuOwnedAllocationSite::topology_vector,
                       allocations))
      {
        return false;
      }
      if (!ResizeOwned(physical_interior_ids,
                       static_cast<std::size_t>(interior_space->GetVSize()),
                       CpuOwnedAllocationSite::topology_vector,
                       allocations)) return false;
      std::fill(physical_interior_ids.begin(), physical_interior_ids.end(),
                std::numeric_limits<std::uint64_t>::max());
      if (local_parent_cell_ids.size() !=
          static_cast<std::size_t>(mesh->GetNE())) return false;
      for (int element = 0; element < mesh->GetNE(); ++element)
      {
        mfem::Array<int> vdofs;
        interior_space->GetElementVDofs(element, vdofs);
        if (vdofs.Size() != local_interior) return false;
        const std::uint64_t parent = local_parent_cell_ids[
            static_cast<std::size_t>(element)];
        for (int scalar = 0; scalar < scalar_interior_dofs; ++scalar)
          for (int component = 0; component < cpu_conserved_components;
               ++component)
          {
            double sign = 1.0;
            const int dof = Decode(vdofs[ByNodes(
                scalar, scalar_interior_dofs, component)], sign);
            if (dof < 0 || dof >= interior_space->GetVSize() || sign != 1.0)
              return false;
            const std::uint64_t canonical =
                (parent * static_cast<std::uint64_t>(scalar_interior_dofs) +
                 static_cast<std::uint64_t>(scalar)) *
                    cpu_conserved_components +
                static_cast<std::uint64_t>(component);
            std::uint64_t& stored = physical_interior_ids[
                static_cast<std::size_t>(dof)];
            if (stored != std::numeric_limits<std::uint64_t>::max() &&
                stored != canonical) return false;
            stored = canonical;
          }
      }
      if (std::any_of(physical_interior_ids.begin(), physical_interior_ids.end(),
                      [](const std::uint64_t value)
                      { return value == std::numeric_limits<std::uint64_t>::max(); }))
      {
        return false;
      }
      for (int local = 0; local < interior_space->GetVSize(); ++local)
      {
        interior_global_ids[static_cast<std::size_t>(local)] =
            static_cast<std::uint64_t>(
                interior_space->GetGlobalTDofNumber(local));
      }
      if (!ResizeOwned(trace_canonical_ids,
                       static_cast<std::size_t>(trace_space->GetTrueVSize()),
                       CpuOwnedAllocationSite::topology_vector,
                       allocations))
      {
        return false;
      }
      std::fill(trace_canonical_ids.begin(), trace_canonical_ids.end(),
                std::numeric_limits<std::uint64_t>::max());
      for (int face = 0; face < mesh->GetNumFaces(); ++face)
      {
        const mfem::FiniteElement* finite_element =
            trace_space->GetFaceElement(face);
        mfem::ElementTransformation* transformation =
            mesh->GetFaceTransformation(face);
        if (!finite_element || !transformation ||
            finite_element->GetDof() != scalar_trace_dofs ||
            scalar_trace_dofs < 2 || scalar_trace_dofs > 3)
        {
          return false;
        }
        mfem::Array<int> vdofs;
        trace_space->GetFaceVDofs(face, vdofs);
        const mfem::IntegrationRule& nodes = finite_element->GetNodes();
        std::array<double, 3> x{}, y{};
        for (int scalar = 0; scalar < scalar_trace_dofs; ++scalar)
        {
          mfem::Vector physical(cpu_dimension);
          transformation->Transform(nodes.IntPoint(scalar), physical);
          x[static_cast<std::size_t>(scalar)] = physical[0];
          y[static_cast<std::size_t>(scalar)] = physical[1];
        }
        const bool vertical =
            std::abs(x[0] - x[static_cast<std::size_t>(
                scalar_trace_dofs - 1)]) < 1.0e-12;
        double fixed = 0.0;
        std::array<double, 3> varying{};
        double minimum_varying = std::numeric_limits<double>::infinity();
        for (int scalar = 0; scalar < scalar_trace_dofs; ++scalar)
        {
          const std::size_t index = static_cast<std::size_t>(scalar);
          fixed += vertical ? x[index] : y[index];
          varying[index] = vertical ? y[index] : x[index];
          minimum_varying = std::min(minimum_varying, varying[index]);
        }
        fixed /= static_cast<double>(scalar_trace_dofs);
        const int fixed_index = PeriodicIndex(
            fixed, config.elements_per_direction);
        const int cell_index = CellIndex(
            minimum_varying, config.elements_per_direction);
        const int face_index = vertical
            ? fixed_index * config.elements_per_direction + cell_index
            : config.elements_per_direction * config.elements_per_direction +
                fixed_index * config.elements_per_direction + cell_index;
        for (int scalar = 0; scalar < scalar_trace_dofs; ++scalar)
        {
          int canonical_scalar = 0;
          for (int other = 0; other < scalar_trace_dofs; ++other)
          {
            if (varying[static_cast<std::size_t>(other)] <
                varying[static_cast<std::size_t>(scalar)])
            {
              ++canonical_scalar;
            }
          }
          for (int component = 0; component < cpu_conserved_components;
               ++component)
          {
            double sign = 1.0;
            const int local_dof = Decode(
                vdofs[ByNodes(scalar, scalar_trace_dofs, component)], sign);
            if (local_dof < 0 ||
                local_dof >= static_cast<int>(
                    exchange_plan->map.dofs.entries.size()))
            {
              return false;
            }
            const trace::TraceMapEntry& entry =
                exchange_plan->map.dofs.entries[
                    static_cast<std::size_t>(local_dof)];
            if (entry.ownership != Ownership::owned) { continue; }
            const std::uint64_t canonical =
                (static_cast<std::uint64_t>(
                     face_index * scalar_trace_dofs + canonical_scalar) *
                     cpu_conserved_components +
                 static_cast<std::uint64_t>(component));
            std::uint64_t& stored = trace_canonical_ids[
                static_cast<std::size_t>(entry.local_true_dof)];
            if (stored != std::numeric_limits<std::uint64_t>::max() &&
                stored != canonical)
            {
              return false;
            }
            stored = canonical;
          }
        }
      }
      return std::none_of(trace_canonical_ids.begin(),
                          trace_canonical_ids.end(),
                          [](const std::uint64_t value)
                          {
                            return value ==
                                std::numeric_limits<std::uint64_t>::max();
                          });
    }
    catch (...)
    {
      return false;
    }
  }

  bool ObserveParentCellIds() noexcept
  {
    bool local_ready = true;
    try
    {
      if (!ResizeOwned(local_parent_cell_ids,
                       static_cast<std::size_t>(mesh->GetNE()),
                       CpuOwnedAllocationSite::topology_vector, allocations))
        local_ready = false;
      for (int element = 0; local_ready && element < mesh->GetNE(); ++element)
      {
        const int attribute = mesh->GetElement(element)->GetAttribute();
        if (attribute <= 0) { local_ready = false; break; }
        local_parent_cell_ids[static_cast<std::size_t>(element)] =
            static_cast<std::uint64_t>(attribute - 1);
        // Attribute values are transient construction provenance.  The
        // numerical mesh retains the physical default after extraction.
        mesh->SetAttribute(element, 1);
      }
      if (!AllRanks(local_ready)) return false;
      mesh->SetAttributes();
      if (mesh->attributes.Size() != 1 || mesh->attributes[0] != 1)
        local_ready = false;
      if (!ResizeOwned(local_parent_cell_ids_sorted,
                       local_parent_cell_ids.size(),
                       CpuOwnedAllocationSite::topology_vector, allocations))
        local_ready = false;
      if (local_ready)
      {
        local_parent_cell_ids_sorted = local_parent_cell_ids;
        std::sort(local_parent_cell_ids_sorted.begin(),
                  local_parent_cell_ids_sorted.end());
      }
      if (local_ready && std::adjacent_find(local_parent_cell_ids_sorted.begin(),
                             local_parent_cell_ids_sorted.end()) !=
          local_parent_cell_ids_sorted.end()) local_ready = false;
      if (!AllRanks(local_ready)) return false;
      const int local_count = mesh->GetNE();
      std::array<int, 2> counts{};
      std::array<int, 2> offsets{};
      const bool counts_gathered = MPI_Allgather(
          &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
          MPI_COMM_WORLD) == MPI_SUCCESS;
      if (!AllRanks(counts_gathered)) return false;
      int total = 0;
      for (int rank = 0; rank < ranks; ++rank)
      {
        if (counts[static_cast<std::size_t>(rank)] <= 0) local_ready = false;
        offsets[static_cast<std::size_t>(rank)] = total;
        total += counts[static_cast<std::size_t>(rank)];
      }
      if (!AllRanks(local_ready)) return false;
      std::vector<std::uint64_t> all;
      try { all.resize(static_cast<std::size_t>(total)); }
      catch (...) { local_ready = false; }
      if (!AllRanks(local_ready)) return false;
      if (MPI_Allgatherv(local_parent_cell_ids.data(), local_count,
                         MPI_UINT64_T, all.data(), counts.data(), offsets.data(),
                         MPI_UINT64_T, MPI_COMM_WORLD) != MPI_SUCCESS)
        return false;
      std::sort(all.begin(), all.end());
      if (total != config.elements_per_direction *
                       config.elements_per_direction) return false;
      for (int element = 0; element < total; ++element)
        if (all[static_cast<std::size_t>(element)] !=
            static_cast<std::uint64_t>(element)) return false;
      return true;
    }
    catch (...)
    {
      return false;
    }
  }

  bool BuildTopology() noexcept
  {
    try
    {
      const int expected_interior_dofs =
          (config.polynomial_order + 1) * (config.polynomial_order + 1);
      scalar_interior_dofs = expected_interior_dofs;
      local_interior = scalar_interior_dofs * interior_components;
      scalar_trace_dofs = config.polynomial_order + 1;
      local_trace = 4 * scalar_trace_dofs *
          trace_components;
      if (mesh->GetNE() > 0)
      {
        const mfem::FiniteElement* interior = interior_space->GetFE(0);
        mfem::Array<int> first_faces, first_orientations;
        mesh->GetElementEdges(0, first_faces, first_orientations);
        if (!interior || interior->GetDof() != expected_interior_dofs ||
            first_faces.Size() != 4)
        {
          return false;
        }
      }
      if (!ResizeOwned(element_interior_vdofs,
                       static_cast<std::size_t>(mesh->GetNE()) *
                           local_interior,
                       CpuOwnedAllocationSite::topology_vector,
                       allocations) ||
          !ResizeOwned(element_faces,
                       static_cast<std::size_t>(mesh->GetNE()) * 4,
                       CpuOwnedAllocationSite::topology_vector,
                       allocations) ||
          !ResizeOwned(shared_faces,
                       static_cast<std::size_t>(mesh->GetNumFaces()),
                       CpuOwnedAllocationSite::topology_vector,
                       allocations) ||
          !ResizeOwned(signed_trace_dofs,
                       static_cast<std::size_t>(mesh->GetNE()) * local_trace,
                       CpuOwnedAllocationSite::topology_vector,
                       allocations))
      {
        return false;
      }
      std::fill(shared_faces.begin(), shared_faces.end(), 0);
      for (int shared_face = 0;
           shared_face < mesh->GetNSharedFaces();
           ++shared_face)
      {
        const int face = mesh->GetSharedFace(shared_face);
        if (face < 0 ||
            face >= static_cast<int>(shared_faces.size()))
        {
          return false;
        }
        shared_faces[static_cast<std::size_t>(face)] = 1;
      }
      for (int element = 0; element < mesh->GetNE(); ++element)
      {
        mfem::Array<int> vdofs;
        interior_space->GetElementVDofs(element, vdofs);
        if (vdofs.Size() != local_interior) { return false; }
        std::copy(vdofs.begin(), vdofs.end(),
                  element_interior_vdofs.begin() +
                      static_cast<std::size_t>(element) * local_interior);
        mfem::Array<int> faces, orientations;
        mesh->GetElementEdges(element, faces, orientations);
        if (faces.Size() != 4) { return false; }
        int trace_offset = 0;
        for (int local_face = 0; local_face < faces.Size(); ++local_face)
        {
          element_faces[static_cast<std::size_t>(element) * 4 + local_face] =
              faces[local_face];
          mfem::Array<int> face_vdofs;
          trace_space->GetFaceVDofs(faces[local_face], face_vdofs);
          if (face_vdofs.Size() !=
              scalar_trace_dofs * trace_components)
          {
            return false;
          }
          std::copy(face_vdofs.begin(), face_vdofs.end(),
                    signed_trace_dofs.begin() +
                        static_cast<std::size_t>(element) * local_trace +
                        trace_offset);
          trace_offset += face_vdofs.Size();
        }
        if (trace_offset != local_trace) { return false; }
      }
      return BuildCanonicalIds();
    }
    catch (...)
    {
      return false;
    }
  }

  bool PrimeCondensationWorkspace() noexcept
  {
    if (mesh->GetNE() == 0)
    {
      current_system.Zero();
      return true;
    }
    current_system.Zero();
    for (int element = 0; element < current_system.elements; ++element)
      for (int index = 0; index < local_interior; ++index)
      {
        current_system.a[
            static_cast<std::size_t>(element) * local_interior *
                local_interior +
            static_cast<std::size_t>(index) * local_interior + index] = 1.0;
      }
    const LocalBlockShape shape{
        local_interior / interior_components,
        local_trace / trace_components,
        interior_components, trace_components};
    const core::LocalSystemBatchView input{
        shape,
        AsConst(View(current_system.a.data(), current_system.elements,
                     local_interior, local_interior)),
        AsConst(View(current_system.b.data(), current_system.elements,
                     local_interior, local_trace)),
        AsConst(View(current_system.c.data(), current_system.elements,
                     local_trace, local_interior)),
        AsConst(View(current_system.d.data(), current_system.elements,
                     local_trace, local_trace)),
        AsConst(View(current_system.f.data(), current_system.elements,
                     local_interior, 1)),
        AsConst(View(current_system.g.data(), current_system.elements,
                     local_trace, 1))};
    const bool primed = condensation.Condense(
        input,
        {View(schur.data(), current_system.elements, local_trace, local_trace),
         View(reduced_rhs.data(), current_system.elements, local_trace, 1)}) ==
        core::Status::success;
    current_system.Zero();
    std::fill(schur.begin(), schur.end(), 0.0);
    std::fill(reduced_rhs.begin(), reduced_rhs.end(), 0.0);
    return primed;
  }

  bool PrimeHypreWorkspace() noexcept
  {
    // This allocation/setup prime is deliberately outside the scientific
    // factor-setup generation and census; every Newton matrix is rebound later.
    bool local_setup_ready = false;
    try
    {
      const auto prepare_prime = [&]() -> bool
      {
        if (!true_matrix || !hypre_rhs || !hypre_solution ||
            !hypre_residual || !ilu || !gmres)
        {
          return false;
        }
        true_matrix->HostReadWrite();
        true_matrix->GetDiag(true_diagonal_view);
        true_matrix->GetOffd(true_offdiagonal_view,
                             true_offdiagonal_columns);
        const HYPRE_BigInt* row_starts = true_matrix->GetRowStarts();
        int* diagonal_i = true_diagonal_view.GetI();
        int* diagonal_j = true_diagonal_view.GetJ();
        double* diagonal_data = true_diagonal_view.GetData();
        double* offdiagonal_data = true_offdiagonal_view.GetData();
        const int diagonal_nnz = true_diagonal_view.NumNonZeroElems();
        const int offdiagonal_nnz = true_offdiagonal_view.NumNonZeroElems();
        const int local_rows = true_diagonal_view.Height();
        // A globally active borrowed mesh may legitimately give this rank no
        // true-trace rows.  Hypre/MFEM then expose empty (and potentially
        // null) local CSR storage; no entry is read or written on that rank.
        if ((local_rows != 0 && !row_starts) ||
            (local_rows != 0 && !diagonal_i) ||
            (diagonal_nnz != 0 && (!diagonal_j || !diagonal_data)) ||
            (offdiagonal_nnz != 0 && !offdiagonal_data))
        {
          return false;
        }
        if (diagonal_nnz != 0)
        {
          std::fill(diagonal_data, diagonal_data + diagonal_nnz, 0.0);
        }
        if (offdiagonal_nnz != 0)
        {
          std::fill(offdiagonal_data, offdiagonal_data + offdiagonal_nnz, 0.0);
        }
        if (local_rows == 0) { return true; }
        const HYPRE_BigInt row_begin = HYPRE_AssumedPartitionCheck()
            ? row_starts[0] : row_starts[rank];
        for (int row = 0; row < local_rows; ++row)
        {
          bool found = false;
          for (int entry = diagonal_i[row]; entry < diagonal_i[row + 1];
               ++entry)
          {
            if (row_begin + diagonal_j[entry] == row_begin + row)
            {
              diagonal_data[entry] = 1.0;
              found = true;
              break;
            }
          }
          if (!found) { return false; }
        }
        return true;
      };
      local_setup_ready = prepare_prime();
    }
    catch (...)
    {
      local_setup_ready = false;
    }
    if (!CollectiveAll(local_setup_ready)) { return false; }

    bool hypre_input_ready = true;
    try
    {
      true_matrix->HypreRead();
      *hypre_rhs = 1.0;
      *hypre_solution = 0.0;
      hypre_input_ready = Finite(*hypre_rhs) && Finite(*hypre_solution) &&
          !ConsumeOwnedCollectiveTestFault(
              detail::CpuCollectiveTestFault::
                  initialization_before_gmres_prime,
              rank, incarnation);
    }
    catch (...)
    {
      hypre_input_ready = false;
    }
    if (!CollectiveAll(hypre_input_ready)) { return false; }

    bool prime_complete = true;
    try
    {
      // This is the opaque-call boundary from the frozen card: all W8-owned
      // preparation was admitted above.  If Mult returns, its result is
      // consensused below.  An in-call interruption is process-terminal; this
      // leaf neither injects it nor fabricates a RuntimeFailure/retry for it.
      gmres->Mult(*hypre_rhs, *hypre_solution);
      prime_complete = Finite(*hypre_solution);
    }
    catch (...)
    {
      prime_complete = false;
    }
    if (!CollectiveAll(prime_complete)) { return false; }

    bool reset_complete = true;
    try
    {
      *hypre_rhs = 0.0;
      *hypre_solution = 0.0;
      *hypre_residual = 0.0;
      reset_complete = Finite(*hypre_rhs) && Finite(*hypre_solution) &&
          Finite(*hypre_residual);
    }
    catch (...)
    {
      reset_complete = false;
    }
    return CollectiveAll(reset_complete);
  }

  bool FindOwnedTrueEntry(const std::uint64_t global_row,
                          const std::uint64_t global_column,
                          int& data_index,
                          bool& offdiagonal) noexcept
  {
    const HYPRE_BigInt* row_starts = true_matrix->GetRowStarts();
    int* diagonal_i = true_diagonal_view.GetI();
    int* diagonal_j = true_diagonal_view.GetJ();
    int* offdiagonal_i = true_offdiagonal_view.GetI();
    int* offdiagonal_j = true_offdiagonal_view.GetJ();
    if (!row_starts || !diagonal_i)
    {
      return false;
    }
    const HYPRE_BigInt row_begin = HYPRE_AssumedPartitionCheck()
        ? row_starts[0] : row_starts[rank];
    const HYPRE_BigInt row_end = row_begin + true_diagonal_view.Height();
    if (global_row < static_cast<std::uint64_t>(row_begin) ||
        global_row >= static_cast<std::uint64_t>(row_end))
    {
      return false;
    }
    const int local_row = static_cast<int>(global_row - row_begin);
    if (global_column >= static_cast<std::uint64_t>(row_begin) &&
        global_column < static_cast<std::uint64_t>(row_end))
    {
      if (!diagonal_j) { return false; }
      const int local_column = static_cast<int>(global_column - row_begin);
      for (int entry = diagonal_i[local_row];
           entry < diagonal_i[local_row + 1]; ++entry)
      {
        if (diagonal_j[entry] == local_column)
        {
          data_index = entry;
          offdiagonal = false;
          return true;
        }
      }
      return false;
    }
    if (!offdiagonal_i || !offdiagonal_j || !true_offdiagonal_columns)
    {
      return false;
    }
    for (int entry = offdiagonal_i[local_row];
         entry < offdiagonal_i[local_row + 1]; ++entry)
    {
      if (static_cast<std::uint64_t>(
              true_offdiagonal_columns[offdiagonal_j[entry]]) ==
          global_column)
      {
        data_index = entry;
        offdiagonal = true;
        return true;
      }
    }
    return false;
  }

  bool BuildSparseSchurPlan() noexcept
  {
    std::size_t owned_count = 0;
    std::size_t outgoing_count = 0;
    std::vector<std::array<std::uint64_t, 2>> outgoing_keys;
    bool outgoing_ready = false;
    try
    {
      const auto prepare_outgoing = [&]() -> bool
      {
      if (!exchange_plan) { return false; }
      for (int element = 0; element < mesh->GetNE(); ++element)
        for (int row = 0; row < local_trace; ++row)
        {
          double sign = 1.0;
          const int row_dof = Decode(signed_trace_dofs[
              static_cast<std::size_t>(element) * local_trace + row], sign);
          if (row_dof < 0 || row_dof >= static_cast<int>(
                  exchange_plan->map.dofs.entries.size()))
          {
            return false;
          }
          const auto& row_entry = exchange_plan->map.dofs.entries[
              static_cast<std::size_t>(row_dof)];
          const std::size_t count = static_cast<std::size_t>(local_trace);
          if (row_entry.owner_rank == rank) { owned_count += count; }
          else { outgoing_count += count; }
        }
      if (!ResizeOwned(owned_schur_terms, owned_count,
                       CpuOwnedAllocationSite::sparse_exchange_graph,
                       allocations) ||
          !ResizeOwned(outgoing_schur_terms, outgoing_count,
                       CpuOwnedAllocationSite::sparse_exchange_graph,
                       allocations))
      {
        return false;
      }
      if (outgoing_count > static_cast<std::size_t>(
              std::numeric_limits<int>::max() / 2) ||
          (ranks == 1 && outgoing_count != 0))
      {
        return false;
      }
      outgoing_keys.resize(outgoing_count);
      std::size_t owned_cursor = 0;
      std::size_t outgoing_cursor = 0;
      for (int element = 0; element < mesh->GetNE(); ++element)
        for (int row = 0; row < local_trace; ++row)
        {
          double row_sign = 1.0;
          const int row_dof = Decode(signed_trace_dofs[
              static_cast<std::size_t>(element) * local_trace + row],
                                     row_sign);
          const auto& row_entry = exchange_plan->map.dofs.entries[
              static_cast<std::size_t>(row_dof)];
          for (int column = 0; column < local_trace; ++column)
          {
            double column_sign = 1.0;
            const int column_dof = Decode(signed_trace_dofs[
                static_cast<std::size_t>(element) * local_trace + column],
                                          column_sign);
            if (column_dof < 0 || column_dof >= static_cast<int>(
                    exchange_plan->map.dofs.entries.size()))
            {
              return false;
            }
            const auto& column_entry = exchange_plan->map.dofs.entries[
                static_cast<std::size_t>(column_dof)];
            const std::size_t schur_index =
                static_cast<std::size_t>(element) * local_trace * local_trace +
                static_cast<std::size_t>(row) * local_trace + column;
            const double sign = row_sign * column_sign;
            if (row_entry.owner_rank == rank)
            {
              int data_index = -1;
              bool offdiagonal = false;
              if (!FindOwnedTrueEntry(row_entry.global_true_dof,
                                      column_entry.global_true_dof,
                                      data_index, offdiagonal))
              {
                return false;
              }
              owned_schur_terms[owned_cursor++] =
                  {schur_index, data_index, sign, offdiagonal};
            }
            else
            {
              outgoing_schur_terms[outgoing_cursor] = {
                  schur_index, sign, row_entry.owner_rank};
              outgoing_keys[outgoing_cursor] = {
                  static_cast<std::uint64_t>(row_entry.global_true_dof),
                  static_cast<std::uint64_t>(column_entry.global_true_dof)};
              ++outgoing_cursor;
            }
          }
        }
      if (owned_cursor != owned_count || outgoing_cursor != outgoing_count)
      {
        return false;
      }

      // Keep a fixed rank-major order.  Alltoallv then gives a deterministic
      // arbitrary-neighbour exchange rather than the former rank-2 peer
      // shortcut.
      std::vector<std::size_t> order(outgoing_count);
      std::iota(order.begin(), order.end(), std::size_t{0});
      std::stable_sort(order.begin(), order.end(),
          [&](const std::size_t left, const std::size_t right)
          { return outgoing_schur_terms[left].destination_rank <
                   outgoing_schur_terms[right].destination_rank; });
      std::vector<SparseOutgoingTerm> ordered_terms(outgoing_count);
      std::vector<std::array<std::uint64_t, 2>> ordered_keys(outgoing_count);
      for (std::size_t index = 0; index < outgoing_count; ++index)
      {
        ordered_terms[index] = outgoing_schur_terms[order[index]];
        ordered_keys[index] = outgoing_keys[order[index]];
      }
      outgoing_schur_terms.swap(ordered_terms);
      outgoing_keys.swap(ordered_keys);
      outgoing_schur_counts.assign(static_cast<std::size_t>(ranks), 0);
      outgoing_schur_offsets.assign(static_cast<std::size_t>(ranks), 0);
      for (const SparseOutgoingTerm& term : outgoing_schur_terms)
      {
        if (term.destination_rank < 0 || term.destination_rank >= ranks ||
            term.destination_rank == rank) return false;
        ++outgoing_schur_counts[static_cast<std::size_t>(term.destination_rank)];
      }
      for (int peer = 1; peer < ranks; ++peer)
        outgoing_schur_offsets[static_cast<std::size_t>(peer)] =
            outgoing_schur_offsets[static_cast<std::size_t>(peer - 1)] +
            outgoing_schur_counts[static_cast<std::size_t>(peer - 1)];

      return !ConsumeOwnedCollectiveTestFault(
          detail::CpuCollectiveTestFault::
              initialization_before_sparse_count_exchange,
          rank, incarnation);
      };
      outgoing_ready = prepare_outgoing();
    }
    catch (...)
    {
      outgoing_ready = false;
    }
    if (!CollectiveAll(outgoing_ready)) { return false; }

    bool count_exchange_ready = true;
    incoming_schur_counts.assign(static_cast<std::size_t>(ranks), 0);
    incoming_schur_offsets.assign(static_cast<std::size_t>(ranks), 0);
    count_exchange_ready = MPI_Alltoall(outgoing_schur_counts.data(), 1,
                                        MPI_INT, incoming_schur_counts.data(),
                                        1, MPI_INT, communicator) == MPI_SUCCESS;
    if (!CollectiveAll(count_exchange_ready)) { return false; }

    std::size_t incoming_count = 0;
    std::vector<std::array<std::uint64_t, 2>> incoming_keys;
    bool incoming_allocation_ready = true;
    try
    {
      for (int peer = 0; peer < ranks; ++peer)
      {
        if (incoming_schur_counts[static_cast<std::size_t>(peer)] < 0)
          incoming_allocation_ready = false;
        if (peer > 0)
          incoming_schur_offsets[static_cast<std::size_t>(peer)] =
              incoming_schur_offsets[static_cast<std::size_t>(peer - 1)] +
              incoming_schur_counts[static_cast<std::size_t>(peer - 1)];
      }
      incoming_count = static_cast<std::size_t>(
          incoming_schur_offsets.back() + incoming_schur_counts.back());
      incoming_allocation_ready = incoming_allocation_ready &&
          incoming_count <= static_cast<std::size_t>(
              std::numeric_limits<int>::max() / 2);
      if (incoming_allocation_ready)
      {
        incoming_keys.resize(incoming_count);
        incoming_allocation_ready = !ConsumeOwnedCollectiveTestFault(
            detail::CpuCollectiveTestFault::
                initialization_between_sparse_exchanges,
            rank, incarnation);
      }
    }
    catch (...)
    {
      incoming_allocation_ready = false;
    }
    if (!CollectiveAll(incoming_allocation_ready)) { return false; }

    std::vector<int> outgoing_key_counts(static_cast<std::size_t>(ranks));
    std::vector<int> incoming_key_counts(static_cast<std::size_t>(ranks));
    std::vector<int> outgoing_key_offsets(static_cast<std::size_t>(ranks));
    std::vector<int> incoming_key_offsets(static_cast<std::size_t>(ranks));
    for (int peer = 0; peer < ranks; ++peer)
    {
      outgoing_key_counts[static_cast<std::size_t>(peer)] =
          2 * outgoing_schur_counts[static_cast<std::size_t>(peer)];
      incoming_key_counts[static_cast<std::size_t>(peer)] =
          2 * incoming_schur_counts[static_cast<std::size_t>(peer)];
      outgoing_key_offsets[static_cast<std::size_t>(peer)] =
          2 * outgoing_schur_offsets[static_cast<std::size_t>(peer)];
      incoming_key_offsets[static_cast<std::size_t>(peer)] =
          2 * incoming_schur_offsets[static_cast<std::size_t>(peer)];
    }
    const bool key_exchange_ready = MPI_Alltoallv(
        outgoing_keys.data(), outgoing_key_counts.data(),
        outgoing_key_offsets.data(), MPI_UINT64_T, incoming_keys.data(),
        incoming_key_counts.data(), incoming_key_offsets.data(), MPI_UINT64_T,
        communicator) == MPI_SUCCESS;
    if (!CollectiveAll(key_exchange_ready)) { return false; }

    bool final_ready = false;
    try
    {
      const auto finalize_plan = [&]() -> bool
      {
        if (!ResizeOwned(incoming_schur_slots, incoming_count,
                         CpuOwnedAllocationSite::sparse_exchange_graph,
                         allocations) ||
            !ResizeOwned(outgoing_schur_values, outgoing_count,
                         CpuOwnedAllocationSite::sparse_exchange_buffer,
                         allocations) ||
            !ResizeOwned(incoming_schur_values, incoming_count,
                         CpuOwnedAllocationSite::sparse_exchange_buffer,
                         allocations))
        {
          return false;
        }
        for (std::size_t index = 0; index < incoming_count; ++index)
        {
          int data_index = -1;
          bool offdiagonal = false;
          if (!FindOwnedTrueEntry(incoming_keys[index][0],
                                  incoming_keys[index][1], data_index,
                                  offdiagonal))
          {
            return false;
          }
          incoming_schur_slots[index] = {data_index, offdiagonal};
        }
        return true;
      };
      final_ready = finalize_plan();
    }
    catch (...)
    {
      final_ready = false;
    }
    return CollectiveAll(final_ready);
  }

  bool AllocateWorkspace() noexcept
  {
    bool local_graph_ready = false;
    try
    {
      const auto prepare_local_graph = [&]() -> bool
      {
      if (!current_system.Allocate(mesh->GetNE(), local_interior, local_trace,
                                   allocations) ||
          !trial_system.Allocate(mesh->GetNE(), local_interior, local_trace,
                                 allocations))
      {
        return false;
      }
      const std::size_t batches = static_cast<std::size_t>(mesh->GetNE());
      const std::size_t ni = static_cast<std::size_t>(local_interior);
      const std::size_t nt = static_cast<std::size_t>(local_trace);
      if (!ResizeOwned(schur, batches * nt * nt,
                       CpuOwnedAllocationSite::condensation_vector,
                       allocations) ||
          !ResizeOwned(reduced_rhs, batches * nt,
                       CpuOwnedAllocationSite::condensation_vector,
                       allocations) ||
          !ResizeOwned(local_trace_increment, batches * nt,
                       CpuOwnedAllocationSite::recovery_vector,
                       allocations) ||
          !ResizeOwned(recovered_interior, batches * ni,
                       CpuOwnedAllocationSite::recovery_vector,
                       allocations) ||
          !ResizeOwned(grid_interior_increment,
                       static_cast<std::size_t>(interior_space->GetVSize()),
                       CpuOwnedAllocationSite::recovery_vector,
                       allocations) ||
          !ResizeOwned(interior_census, grid_interior_increment.size(),
                       CpuOwnedAllocationSite::recovery_vector,
                       allocations))
      {
        return false;
      }

      old_interior = std::make_unique<mfem::ParGridFunction>(
          interior_space.get());
      external_source = std::make_unique<mfem::ParGridFunction>(
          interior_space.get());
      iterate_interior = std::make_unique<mfem::ParGridFunction>(
          interior_space.get());
      trial_interior = std::make_unique<mfem::ParGridFunction>(
          interior_space.get());
      iterate_trace = std::make_unique<mfem::ParGridFunction>(trace_space.get());
      trial_trace = std::make_unique<mfem::ParGridFunction>(trace_space.get());
      if (!ObserveWrapper(old_interior) || !ObserveWrapper(external_source) ||
          !ObserveWrapper(iterate_interior) ||
          !ObserveWrapper(trial_interior) || !ObserveWrapper(iterate_trace) ||
          !ObserveWrapper(trial_trace))
      {
        return false;
      }
      iterate_true_trace.SetSize(trace_space->GetTrueVSize());
      trial_true_trace.SetSize(trace_space->GetTrueVSize());
      true_increment.SetSize(trace_space->GetTrueVSize());
      trace_distribution_workspace.SetSize(trace_space->GetVSize());
      distributed_increment.SetSize(trace_space->GetVSize());
      local_trace_rhs.SetSize(trace_space->GetVSize());
      true_trace_rhs.SetSize(trace_space->GetTrueVSize());
      local_trace_residual.SetSize(trace_space->GetVSize());
      true_trace_residual.SetSize(trace_space->GetTrueVSize());
      volume_shape_workspace.SetSize(scalar_interior_dofs);
      volume_derivative_workspace.SetSize(scalar_interior_dofs,
                                          cpu_dimension);
      physical_workspace.SetSize(cpu_dimension);
      face_interior_shape_workspace.SetSize(scalar_interior_dofs);
      face_trace_shape_workspace.SetSize(scalar_trace_dofs);
      scaled_normal_workspace.SetSize(cpu_dimension);
      element_trace_dofs_workspace.SetSize(local_trace);
      local_schur_workspace.SetSize(local_trace);
      if (mfem::IntRules.Get(mfem::Geometry::SQUARE,
                             config.assembly_quadrature_order).GetNPoints() <= 0 ||
          mfem::IntRules.Get(mfem::Geometry::SEGMENT,
                             config.assembly_quadrature_order).GetNPoints() <= 0)
      {
        return false;
      }
      local_matrix = std::make_unique<mfem::SparseMatrix>(
          trace_space->GetVSize());
      if (!ObserveWrapper(local_matrix)) { return false; }
      mfem::DenseMatrix graph_pattern(local_trace);
      for (int element = 0; element < mesh->GetNE(); ++element)
      {
        mfem::Array<int> dofs(local_trace);
        for (int row = 0; row < local_trace; ++row)
        {
          const int signed_row = signed_trace_dofs[
              static_cast<std::size_t>(element) * local_trace + row];
          double row_sign = 1.0;
          (void)Decode(signed_row, row_sign);
          dofs[row] = signed_row;
          for (int column = 0; column < local_trace; ++column)
          {
            double column_sign = 1.0;
            (void)Decode(signed_trace_dofs[
                static_cast<std::size_t>(element) * local_trace + column],
                column_sign);
            // AddSubMatrix applies the signed-VDof row/column signs.  Feeding
            // their product here yields a strictly positive symbolic graph.
            graph_pattern(row, column) = row_sign * column_sign;
          }
        }
        local_matrix->AddSubMatrix(dofs, dofs, graph_pattern, 1);
      }
      local_matrix->Finalize(0, false);
      parallel_form =
          std::make_unique<mfem::ParBilinearForm>(trace_space.get());
      if (!ObserveWrapper(parallel_form)) { return false; }
      return true;
      };
      local_graph_ready = prepare_local_graph();
    }
    catch (...)
    {
      local_graph_ready = false;
    }
    if (!CollectiveAll(local_graph_ready)) { return false; }

    bool parallel_assembly_ready = false;
    try
    {
      // ParallelAssemble is opaque.  The complete W8-owned graph was admitted
      // above, and every returned result is consensused below.  No in-call
      // failure seam or same-process recovery claim exists here.
      true_matrix.reset(parallel_form->ParallelAssemble(local_matrix.get()));
      parallel_assembly_ready = ObserveWrapper(true_matrix) && true_matrix &&
          true_matrix->Height() == trace_space->GetTrueVSize() &&
          true_matrix->GetGlobalNumRows() == trace_space->GlobalTrueVSize() &&
          true_matrix->GetGlobalNumCols() == trace_space->GlobalTrueVSize();
      if (parallel_assembly_ready)
      {
        true_matrix->GetDiag(true_diagonal_view);
        true_matrix->GetOffd(true_offdiagonal_view,
                             true_offdiagonal_columns);
      }
    }
    catch (...)
    {
      parallel_assembly_ready = false;
    }
    if (!CollectiveAll(parallel_assembly_ready)) { return false; }
    if (!BuildSparseSchurPlan()) { return false; }

    bool condensation_prime_ready = true;
    try
    {
      *local_matrix = 0.0;
      condensation_prime_ready = PrimeCondensationWorkspace();
    }
    catch (...)
    {
      condensation_prime_ready = false;
    }
    if (!CollectiveAll(condensation_prime_ready)) { return false; }

    bool hypre_rhs_ready = false;
    try
    {
      hypre_rhs = std::make_unique<mfem::HypreParVector>(trace_space.get());
      hypre_rhs_ready = ObserveWrapper(hypre_rhs);
    }
    catch (...)
    {
      hypre_rhs_ready = false;
    }
    if (!CollectiveAll(hypre_rhs_ready)) { return false; }

    bool hypre_solution_ready = false;
    try
    {
      hypre_solution =
          std::make_unique<mfem::HypreParVector>(trace_space.get());
      hypre_solution_ready = ObserveWrapper(hypre_solution);
    }
    catch (...)
    {
      hypre_solution_ready = false;
    }
    if (!CollectiveAll(hypre_solution_ready)) { return false; }

    bool hypre_residual_ready = false;
    try
    {
      hypre_residual =
          std::make_unique<mfem::HypreParVector>(trace_space.get());
      hypre_residual_ready = ObserveWrapper(hypre_residual);
    }
    catch (...)
    {
      hypre_residual_ready = false;
    }
    if (!CollectiveAll(hypre_residual_ready)) { return false; }

    bool ilu_ready = false;
    try
    {
      ilu = std::make_unique<mfem::HypreILU>();
      ApplyFrozenIluSettings();
      ilu_ready = ObserveWrapper(ilu);
    }
    catch (...)
    {
      ilu_ready = false;
    }
    if (!CollectiveAll(ilu_ready)) { return false; }

    bool gmres_ready = false;
    try
    {
      gmres = std::make_unique<mfem::HypreGMRES>(communicator);
      gmres_ready = ObserveWrapper(gmres);
    }
    catch (...)
    {
      gmres_ready = false;
    }
    if (!CollectiveAll(gmres_ready)) { return false; }

    bool solver_configuration_ready = true;
    try
    {
      gmres->SetTol(config.linear_relative_tolerance);
      gmres->SetAbsTol(0.0);
      const int global_trace_size =
          static_cast<int>(trace_space->GlobalTrueVSize());
      const int maximum_iterations =
          generic::ResolveEulerHdgMaximumLinearIterations(
              config.maximum_linear_iterations, global_trace_size);
      gmres->SetMaxIter(maximum_iterations);
      gmres->SetKDim(
          generic::EulerHdgKrylovRestartDimension(global_trace_size));
      gmres->SetLogging(1);
      gmres->SetPrintLevel(0);
    }
    catch (...)
    {
      solver_configuration_ready = false;
    }
    if (!CollectiveAll(solver_configuration_ready)) { return false; }

    bool preconditioner_binding_ready = true;
    try
    {
      gmres->SetPreconditioner(*ilu);
    }
    catch (...)
    {
      preconditioner_binding_ready = false;
    }
    if (!CollectiveAll(preconditioner_binding_ready)) { return false; }

    bool operator_binding_ready = true;
    try
    {
      gmres->SetOperator(*true_matrix);
    }
    catch (...)
    {
      operator_binding_ready = false;
    }
    if (!CollectiveAll(operator_binding_ready)) { return false; }
    return PrimeHypreWorkspace();
  }

  std::size_t WorkspaceBytes() const noexcept
  {
    const auto bytes = [](const auto& vector)
    { return vector.capacity() * sizeof(typename std::decay_t<decltype(vector)>::value_type); };
    return bytes(current_system.residual_interior) +
        bytes(current_system.residual_trace) + bytes(current_system.a) +
        bytes(current_system.b) + bytes(current_system.c) +
        bytes(current_system.d) + bytes(current_system.f) +
        bytes(current_system.g) + bytes(trial_system.residual_interior) +
        bytes(trial_system.residual_trace) + bytes(trial_system.a) +
        bytes(trial_system.b) + bytes(trial_system.c) +
        bytes(trial_system.d) + bytes(trial_system.f) +
        bytes(trial_system.g) + bytes(schur) + bytes(reduced_rhs) +
        bytes(local_trace_increment) + bytes(recovered_interior) +
        bytes(grid_interior_increment) + bytes(interior_census) +
        bytes(owned_schur_terms) + bytes(outgoing_schur_terms) +
        bytes(incoming_schur_slots) + bytes(outgoing_schur_values) +
        bytes(incoming_schur_values);
  }

  bool Assemble(const mfem::ParGridFunction& current_interior,
                const mfem::ParGridFunction& current_trace,
                LocalSystem& system, const int quadrature_order,
                const bool consume_internal_test_hooks = true,
                const std::vector<double>* const frozen_borrowed_llf = nullptr,
                std::vector<double>* const evaluated_borrowed_llf = nullptr)
      noexcept
  {
    system.Zero();
    try
    {
      const bool regularized = IsIgr();
      if ((frozen_borrowed_llf || evaluated_borrowed_llf) &&
          (!IsBorrowed() || regularized))
      {
        return false;
      }
      if (evaluated_borrowed_llf) { evaluated_borrowed_llf->clear(); }
      std::size_t borrowed_llf_index = 0;
      const double alpha = regularized
          ? borrowed_request->numerical_policy.stabilization.alpha : 0.0;
      const double beta = regularized
          ? borrowed_request->numerical_policy.stabilization.beta : 0.0;
      if (regularized &&
          (!Finite(alpha) || !(alpha > 0.0) ||
           !Finite(beta) || !(beta > 0.0)))
      {
        return false;
      }

      for (int element = 0; element < mesh->GetNE(); ++element)
      {
        const mfem::FiniteElement* finite_element =
            interior_space->GetFE(element);
        mfem::ElementTransformation* transformation =
            mesh->GetElementTransformation(element);
        if (!finite_element || !transformation ||
            finite_element->GetDof() != scalar_interior_dofs)
        {
          return false;
        }
        const int* interior_vdofs = element_interior_vdofs.data() +
            static_cast<std::size_t>(element) * local_interior;
        const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
            finite_element->GetGeomType(), quadrature_order);
        const std::size_t element_ni =
            static_cast<std::size_t>(element) * local_interior;
        const std::size_t element_nt =
            static_cast<std::size_t>(element) * local_trace;
        const std::size_t element_a = element_ni * local_interior;
        const std::size_t element_b =
            static_cast<std::size_t>(element) * local_interior * local_trace;
        const std::size_t element_c =
            static_cast<std::size_t>(element) * local_trace * local_interior;
        const std::size_t element_d =
            static_cast<std::size_t>(element) * local_trace * local_trace;

        for (int q_index = 0; q_index < volume_rule.GetNPoints(); ++q_index)
        {
          const mfem::IntegrationPoint& point =
              volume_rule.IntPoint(q_index);
          transformation->SetIntPoint(&point);
          transformation->Transform(point, physical_workspace);
          finite_element->CalcShape(point, volume_shape_workspace);
          finite_element->CalcPhysDShape(*transformation,
                                         volume_derivative_workspace);

          State state{}, old_state{}, source{}, flux_x{}, flux_y{};
          Matrix45 jacobian_x{}, jacobian_y{};
          if (!Interpolate(current_interior, interior_vdofs,
                           scalar_interior_dofs, volume_shape_workspace,
                           state) ||
              !Interpolate(*old_interior, interior_vdofs,
                           scalar_interior_dofs, volume_shape_workspace,
                           old_state) ||
              !(IsBorrowed()
                    ? external_source && Interpolate(
                          *external_source, interior_vdofs,
                          scalar_interior_dofs, volume_shape_workspace, source)
                    : IsW8cConfig(config)
                    ? (source = State{}, true)
                    : ManufacturedSource(physical_workspace[0],
                                         physical_workspace[1], source)))
          {
            return false;
          }

          double sigma = 0.0;
          std::array<double, 2> z{};
          std::array<std::array<double, 2>, 3> conservative_gradient{};
          if (regularized)
          {
            if (!InterpolateComponent(current_interior, interior_vdofs,
                                      scalar_interior_dofs,
                                      volume_shape_workspace, 4, sigma) ||
                !InterpolateComponent(current_interior, interior_vdofs,
                                      scalar_interior_dofs,
                                      volume_shape_workspace, 5, z[0]) ||
                !InterpolateComponent(current_interior, interior_vdofs,
                                      scalar_interior_dofs,
                                      volume_shape_workspace, 6, z[1]))
            {
              return false;
            }
            for (int component = 0; component < 3; ++component)
              for (int direction = 0; direction < cpu_dimension; ++direction)
                if (!InterpolateDerivative(
                        current_interior, interior_vdofs,
                        scalar_interior_dofs, volume_derivative_workspace,
                        component, direction,
                        conservative_gradient[component][direction]))
                {
                  return false;
                }
            if (!RegularizedNormalFlux(state, sigma, {{1.0, 0.0}},
                                       flux_x, jacobian_x) ||
                !RegularizedNormalFlux(state, sigma, {{0.0, 1.0}},
                                       flux_y, jacobian_y))
            {
              return false;
            }
          }
          else
          {
            Matrix4 conventional_x{}, conventional_y{};
            if (!NormalFlux(state, {{1.0, 0.0}}, flux_x) ||
                !NormalFlux(state, {{0.0, 1.0}}, flux_y) ||
                !NormalFluxJacobian(state, {{1.0, 0.0}},
                                    conventional_x) ||
                !NormalFluxJacobian(state, {{0.0, 1.0}},
                                    conventional_y))
            {
              return false;
            }
            for (int row = 0; row < cpu_conserved_components; ++row)
              for (int column = 0;
                   column < cpu_conserved_components; ++column)
              {
                jacobian_x[row][column] = conventional_x[row][column];
                jacobian_y[row][column] = conventional_y[row][column];
              }
          }

          const double weight = point.weight * transformation->Weight();
          if (!Finite(weight) || !(weight > 0.0)) { return false; }

          for (int row_scalar = 0;
               row_scalar < scalar_interior_dofs; ++row_scalar)
          {
            for (int equation = 0;
                 equation < cpu_conserved_components; ++equation)
            {
              const int row =
                  ByNodes(row_scalar, scalar_interior_dofs, equation);
              system.residual_interior[element_ni + row] += weight *
                  (volume_shape_workspace[row_scalar] *
                       (state[equation] - old_state[equation]) / config.dt -
                   volume_derivative_workspace(row_scalar, 0) *
                       flux_x[equation] -
                   volume_derivative_workspace(row_scalar, 1) *
                       flux_y[equation] -
                   volume_shape_workspace[row_scalar] * source[equation]);

              const int euler_columns =
                  regularized ? 5 : cpu_conserved_components;
              for (int column_scalar = 0;
                   column_scalar < scalar_interior_dofs; ++column_scalar)
                for (int variable = 0; variable < euler_columns; ++variable)
                {
                  const int column = ByNodes(
                      column_scalar, scalar_interior_dofs, variable);
                  const double mass = variable == equation
                      ? volume_shape_workspace[row_scalar] *
                            volume_shape_workspace[column_scalar] / config.dt
                      : 0.0;
                  const double volume =
                      -(volume_derivative_workspace(row_scalar, 0) *
                            jacobian_x[equation][variable] +
                        volume_derivative_workspace(row_scalar, 1) *
                            jacobian_y[equation][variable]) *
                      volume_shape_workspace[column_scalar];
                  system.a[element_a +
                           static_cast<std::size_t>(row) * local_interior +
                           column] += weight * (mass + volume);
                }
            }

            if (!regularized) { continue; }

            for (int direction = 0;
                 direction < cpu_dimension; ++direction)
            {
              const int row = ByNodes(
                  row_scalar, scalar_interior_dofs, 4 + direction);
              system.residual_interior[element_ni + row] += weight *
                  (volume_shape_workspace[row_scalar] * state[0] *
                       z[direction] +
                   volume_derivative_workspace(row_scalar, direction) *
                       sigma);
              for (int column_scalar = 0;
                   column_scalar < scalar_interior_dofs; ++column_scalar)
              {
                const double column_shape =
                    volume_shape_workspace[column_scalar];
                system.a[element_a +
                    static_cast<std::size_t>(row) * local_interior +
                    ByNodes(column_scalar, scalar_interior_dofs, 0)] +=
                    weight * volume_shape_workspace[row_scalar] *
                    z[direction] * column_shape;
                system.a[element_a +
                    static_cast<std::size_t>(row) * local_interior +
                    ByNodes(column_scalar, scalar_interior_dofs, 4)] +=
                    weight *
                    volume_derivative_workspace(row_scalar, direction) *
                    column_shape;
                system.a[element_a +
                    static_cast<std::size_t>(row) * local_interior +
                    ByNodes(column_scalar, scalar_interior_dofs,
                            5 + direction)] +=
                    weight * volume_shape_workspace[row_scalar] * state[0] *
                    column_shape;
              }
            }

            std::array<std::array<double, 2>, 2> velocity_gradient{};
            for (int velocity = 0; velocity < cpu_dimension; ++velocity)
              for (int direction = 0;
                   direction < cpu_dimension; ++direction)
              {
                velocity_gradient[velocity][direction] =
                    conservative_gradient[1 + velocity][direction] /
                        state[0] -
                    state[1 + velocity] *
                        conservative_gradient[0][direction] /
                        (state[0] * state[0]);
              }
            const double velocity_trace =
                velocity_gradient[0][0] + velocity_gradient[1][1];
            const double q =
                velocity_gradient[0][0] * velocity_gradient[0][0] +
                2.0 * velocity_gradient[0][1] *
                    velocity_gradient[1][0] +
                velocity_gradient[1][1] * velocity_gradient[1][1] +
                velocity_trace * velocity_trace;
            std::array<std::array<double, 2>, 2> dq_dgradient{};
            for (int velocity = 0; velocity < cpu_dimension; ++velocity)
              for (int direction = 0;
                   direction < cpu_dimension; ++direction)
                dq_dgradient[velocity][direction] =
                    2.0 * velocity_gradient[direction][velocity] +
                    (velocity == direction
                         ? 2.0 * velocity_trace : 0.0);

            const int sigma_row =
                ByNodes(row_scalar, scalar_interior_dofs, 6);
            system.residual_interior[element_ni + sigma_row] += weight *
                (volume_shape_workspace[row_scalar] * sigma / state[0] +
                 alpha *
                     (volume_derivative_workspace(row_scalar, 0) * z[0] +
                      volume_derivative_workspace(row_scalar, 1) * z[1]) -
                 alpha * volume_shape_workspace[row_scalar] * q);

            for (int column_scalar = 0;
                 column_scalar < scalar_interior_dofs; ++column_scalar)
            {
              const double column_shape =
                  volume_shape_workspace[column_scalar];
              system.a[element_a +
                  static_cast<std::size_t>(sigma_row) * local_interior +
                  ByNodes(column_scalar, scalar_interior_dofs, 4)] +=
                  weight * volume_shape_workspace[row_scalar] *
                  column_shape / state[0];
              for (int direction = 0;
                   direction < cpu_dimension; ++direction)
                system.a[element_a +
                    static_cast<std::size_t>(sigma_row) * local_interior +
                    ByNodes(column_scalar, scalar_interior_dofs,
                            5 + direction)] +=
                    weight * alpha *
                    volume_derivative_workspace(row_scalar, direction) *
                    column_shape;

              for (int variable = 0;
                   variable < cpu_conserved_components; ++variable)
              {
                double dq = 0.0;
                if (variable < 3)
                  for (int velocity = 0;
                       velocity < cpu_dimension; ++velocity)
                    for (int direction = 0;
                         direction < cpu_dimension; ++direction)
                    {
                      const double momentum = state[1 + velocity];
                      const double momentum_derivative =
                          conservative_gradient[1 + velocity][direction];
                      const double density_derivative =
                          conservative_gradient[0][direction];
                      const double delta_density =
                          variable == 0 ? column_shape : 0.0;
                      const double delta_density_derivative =
                          variable == 0
                              ? volume_derivative_workspace(
                                    column_scalar, direction)
                              : 0.0;
                      const double delta_momentum =
                          variable == 1 + velocity ? column_shape : 0.0;
                      const double delta_momentum_derivative =
                          variable == 1 + velocity
                              ? volume_derivative_workspace(
                                    column_scalar, direction)
                              : 0.0;
                      const double delta_gradient =
                          delta_momentum_derivative / state[0] -
                          momentum_derivative * delta_density /
                              (state[0] * state[0]) -
                          delta_momentum * density_derivative /
                              (state[0] * state[0]) -
                          momentum * delta_density_derivative /
                              (state[0] * state[0]) +
                          2.0 * momentum * density_derivative *
                              delta_density /
                              (state[0] * state[0] * state[0]);
                      dq += dq_dgradient[velocity][direction] *
                          delta_gradient;
                    }
                double derivative =
                    -alpha * volume_shape_workspace[row_scalar] * dq;
                if (variable == 0)
                  derivative -= volume_shape_workspace[row_scalar] * sigma *
                      column_shape / (state[0] * state[0]);
                system.a[element_a +
                    static_cast<std::size_t>(sigma_row) * local_interior +
                    ByNodes(column_scalar, scalar_interior_dofs,
                            variable)] += weight * derivative;
              }
            }
          }
        }

        int trace_offset = 0;
        for (int local_face = 0; local_face < 4; ++local_face)
        {
          const int face = element_faces[
              static_cast<std::size_t>(element) * 4 + local_face];
          mfem::FaceElementTransformations* face_transformation =
              mesh->GetFaceElementTransformations(face);
          const mfem::FiniteElement* trace_element =
              trace_space->GetFaceElement(face);
          if (!face_transformation || !face_transformation->Face ||
              !trace_element || trace_element->GetDof() != scalar_trace_dofs)
          {
            return false;
          }
          const bool true_boundary =
              face >= 0 &&
              face < static_cast<int>(shared_faces.size()) &&
              shared_faces[static_cast<std::size_t>(face)] == 0 &&
              face_transformation->Elem2No < 0;
          const bool element_one = face_transformation->Elem1No == element;
          if (!element_one && face_transformation->Elem2No != element)
          {
            return false;
          }
          const int* face_trace_vdofs = signed_trace_dofs.data() +
              static_cast<std::size_t>(element) * local_trace + trace_offset;
          const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
              trace_element->GetGeomType(), quadrature_order);

          double face_volume = 0.0;
          if (regularized)
          {
            for (int measure_q = 0;
                 measure_q < face_rule.GetNPoints(); ++measure_q)
            {
              const mfem::IntegrationPoint& measure_point =
                  face_rule.IntPoint(measure_q);
              face_transformation->SetAllIntPoints(&measure_point);
              const double measure =
                  measure_point.weight * face_transformation->Face->Weight();
              if (!Finite(measure) || !(measure > 0.0)) { return false; }
              face_volume += measure;
            }
          }
          const double element_volume =
              regularized ? mesh->GetElementVolume(element) : 1.0;
          const double h = regularized
              ? element_volume / face_volume : 1.0;
          const double degree_scale =
              static_cast<double>((config.polynomial_order + 1) *
                                  (config.polynomial_order + 1));
          if (regularized &&
              (!Finite(element_volume) || !(element_volume > 0.0) ||
               !Finite(face_volume) || !(face_volume > 0.0) ||
               !Finite(h) || !(h > 0.0)))
          {
            return false;
          }

          for (int q_index = 0;
               q_index < face_rule.GetNPoints(); ++q_index)
          {
            const mfem::IntegrationPoint& face_point =
                face_rule.IntPoint(q_index);
            face_transformation->SetAllIntPoints(&face_point);
            const mfem::IntegrationPoint& element_point = element_one
                ? face_transformation->GetElement1IntPoint()
                : face_transformation->GetElement2IntPoint();
            mfem::CalcOrtho(face_transformation->Face->Jacobian(),
                            scaled_normal_workspace);
            const double normal_norm = scaled_normal_workspace.Norml2();
            const double face_measure =
                face_transformation->Face->Weight();
            if (!Finite(normal_norm) || !(normal_norm > 0.0) ||
                !Finite(face_measure) || !(face_measure > 0.0))
            {
              return false;
            }
            const double side = element_one ? 1.0 : -1.0;
            const std::array<double, 2> normal{{
                side * scaled_normal_workspace[0] / normal_norm,
                side * scaled_normal_workspace[1] / normal_norm}};
            finite_element->CalcShape(element_point,
                                      face_interior_shape_workspace);
            trace_element->CalcShape(face_point,
                                     face_trace_shape_workspace);

            State interior_state{}, trace_state{}, trace_flux{},
                exterior_state{};
            Matrix45 trace_jacobian{};
            if (!Interpolate(current_interior, interior_vdofs,
                             scalar_interior_dofs,
                             face_interior_shape_workspace,
                             interior_state) ||
                !Interpolate(current_trace, face_trace_vdofs,
                             scalar_trace_dofs,
                             face_trace_shape_workspace,
                             trace_state))
            {
              return false;
            }

            double sigma = 0.0;
            double sigma_hat = 0.0;
            std::array<double, 2> z{};
            if (regularized)
            {
              if (!InterpolateComponent(
                      current_interior, interior_vdofs,
                      scalar_interior_dofs,
                      face_interior_shape_workspace, 4, sigma) ||
                  !InterpolateComponent(
                      current_interior, interior_vdofs,
                      scalar_interior_dofs,
                      face_interior_shape_workspace, 5, z[0]) ||
                  !InterpolateComponent(
                      current_interior, interior_vdofs,
                      scalar_interior_dofs,
                      face_interior_shape_workspace, 6, z[1]) ||
                  !InterpolateComponent(
                      current_trace, face_trace_vdofs,
                      scalar_trace_dofs,
                      face_trace_shape_workspace, 4, sigma_hat) ||
                  !RegularizedNormalFlux(trace_state, sigma_hat, normal,
                                         trace_flux, trace_jacobian))
              {
                return false;
              }
            }
            else
            {
              Matrix4 conventional_jacobian{};
              if (!NormalFlux(trace_state, normal, trace_flux) ||
                  !NormalFluxJacobian(trace_state, normal,
                                      conventional_jacobian))
              {
                return false;
              }
              for (int row = 0; row < cpu_conserved_components; ++row)
                for (int column = 0;
                     column < cpu_conserved_components; ++column)
                  trace_jacobian[row][column] =
                      conventional_jacobian[row][column];
            }

            double stabilization = beta;
            if (!regularized)
            {
              if (IsBorrowed())
              {
                double evaluated_stabilization = 0.0;
                if (!BorrowedEulerLlfStabilization(
                        interior_state, trace_state, normal,
                        evaluated_stabilization))
                {
                  return false;
                }
                if (evaluated_borrowed_llf)
                  evaluated_borrowed_llf->push_back(
                      evaluated_stabilization);
                if (frozen_borrowed_llf)
                {
                  if (borrowed_llf_index >= frozen_borrowed_llf->size() ||
                      !Finite((*frozen_borrowed_llf)[borrowed_llf_index]) ||
                      !((*frozen_borrowed_llf)[borrowed_llf_index] > 0.0))
                  {
                    return false;
                  }
                  stabilization =
                      (*frozen_borrowed_llf)[borrowed_llf_index];
                }
                else
                {
                  stabilization = evaluated_stabilization;
                }
                ++borrowed_llf_index;
                // Evaluate LLF at this residual candidate.  Its value is
                // deliberately frozen in A/B/C/D below: Newton and line
                // search candidates refresh it only through a new Assemble
                // call, never through a spectral-radius derivative.
              }
              else
              {
                stabilization = cpu_fixed_tau;
              }
            }

            bool physical_boundary = false;
            const int boundary_element = mesh->GetFaceToBdrElMap()[face];
            if (IsBorrowed() && true_boundary && boundary_element >= 0)
            {
              mfem::Vector position(cpu_dimension);
              face_transformation->Face->Transform(face_point, position);
              const int attribute = mesh->GetBdrAttribute(boundary_element);
              const generic::BoundaryPointView boundary_point{
                  position.HostRead(), normal.data(), cpu_dimension,
                  attribute};
              if (attribute <= 0 ||
                  !borrowed_request->boundary.exterior_state(
                      borrowed_request->boundary.context, boundary_point,
                      interior_state.data(), exterior_state.data()) ||
                  !Finite(exterior_state))
              {
                return false;
              }
              physical_boundary = true;
              if (consume_internal_test_hooks)
              {
                ++generic_boundary_picard_test_state.callbacks;
                ++generic_boundary_picard_test_state.physical_rows;
              }
            }

            double tau = 0.0;
            double jump = 0.0;
            double z_hat_normal = 0.0;
            if (regularized)
            {
              if (!(trace_state[0] > density_gate)) { return false; }
              tau = degree_scale / (trace_state[0] * h);
              jump = sigma - sigma_hat;
              z_hat_normal = z[0] * normal[0] + z[1] * normal[1] -
                  tau * jump;
              if (!Finite(tau) || !(tau > 0.0) ||
                  !Finite(z_hat_normal))
              {
                return false;
              }
            }

            const double weight = face_point.weight * face_measure;
            for (int row_scalar = 0;
                 row_scalar < scalar_interior_dofs; ++row_scalar)
              for (int equation = 0;
                   equation < cpu_conserved_components; ++equation)
              {
                const int row =
                    ByNodes(row_scalar, scalar_interior_dofs, equation);
                const double numerical_flux = trace_flux[equation] +
                    stabilization *
                        (interior_state[equation] - trace_state[equation]);
                system.residual_interior[element_ni + row] +=
                    weight * face_interior_shape_workspace[row_scalar] *
                    numerical_flux;
              }

            for (int row_scalar = 0;
                 row_scalar < scalar_trace_dofs; ++row_scalar)
              for (int equation = 0;
                   equation < cpu_conserved_components; ++equation)
              {
                const int row = trace_offset +
                    ByNodes(row_scalar, scalar_trace_dofs, equation);
                const double numerical_flux = trace_flux[equation] +
                    stabilization *
                        (interior_state[equation] - trace_state[equation]);
                system.residual_trace[element_nt + row] +=
                    weight * face_trace_shape_workspace[row_scalar] *
                    (physical_boundary
                         ? trace_state[equation] -
                               exterior_state[equation]
                         : numerical_flux);
              }

            for (int row_scalar = 0;
                 row_scalar < scalar_interior_dofs; ++row_scalar)
              for (int column_scalar = 0;
                   column_scalar < scalar_interior_dofs; ++column_scalar)
                for (int row_component = 0;
                     row_component < cpu_conserved_components;
                     ++row_component)
                  for (int column_component = 0;
                       column_component < cpu_conserved_components;
                       ++column_component)
                  {
                    const int row = ByNodes(
                        row_scalar, scalar_interior_dofs, row_component);
                    const int column = ByNodes(
                        column_scalar, scalar_interior_dofs,
                        column_component);
                    const double derivative =
                        row_component == column_component
                            ? stabilization : 0.0;
                    system.a[element_a +
                        static_cast<std::size_t>(row) * local_interior +
                        column] += weight *
                        face_interior_shape_workspace[row_scalar] *
                        face_interior_shape_workspace[column_scalar] *
                        derivative;
                  }

            const int euler_trace_columns =
                regularized ? 5 : cpu_conserved_components;
            for (int row_scalar = 0;
                 row_scalar < scalar_interior_dofs; ++row_scalar)
              for (int column_scalar = 0;
                   column_scalar < scalar_trace_dofs; ++column_scalar)
                for (int equation = 0;
                     equation < cpu_conserved_components; ++equation)
                  for (int variable = 0;
                       variable < euler_trace_columns; ++variable)
                  {
                    const int row =
                        ByNodes(row_scalar, scalar_interior_dofs, equation);
                    const int column = trace_offset +
                        ByNodes(column_scalar, scalar_trace_dofs, variable);
                    const double derivative =
                        trace_jacobian[equation][variable] -
                        (variable == equation ? stabilization : 0.0);
                    system.b[element_b +
                        static_cast<std::size_t>(row) * local_trace +
                        column] += weight *
                        face_interior_shape_workspace[row_scalar] *
                        face_trace_shape_workspace[column_scalar] *
                        derivative;
                  }

            for (int row_scalar = 0;
                 row_scalar < scalar_trace_dofs; ++row_scalar)
              for (int column_scalar = 0;
                   column_scalar < scalar_interior_dofs; ++column_scalar)
                for (int row_component = 0;
                     row_component < cpu_conserved_components;
                     ++row_component)
                  for (int column_component = 0;
                       column_component < cpu_conserved_components;
                       ++column_component)
                  {
                    const int row = trace_offset + ByNodes(
                        row_scalar, scalar_trace_dofs, row_component);
                    const int column = ByNodes(
                        column_scalar, scalar_interior_dofs,
                        column_component);
                    const double derivative = physical_boundary
                        ? 0.0
                        : (row_component == column_component
                               ? stabilization : 0.0);
                    system.c[element_c +
                        static_cast<std::size_t>(row) * local_interior +
                        column] += weight *
                        face_trace_shape_workspace[row_scalar] * derivative *
                        face_interior_shape_workspace[column_scalar];
                  }

            for (int row_scalar = 0;
                 row_scalar < scalar_trace_dofs; ++row_scalar)
              for (int column_scalar = 0;
                   column_scalar < scalar_trace_dofs; ++column_scalar)
                for (int equation = 0;
                     equation < cpu_conserved_components; ++equation)
                  for (int variable = 0;
                       variable < euler_trace_columns; ++variable)
                  {
                    const int row = trace_offset +
                        ByNodes(row_scalar, scalar_trace_dofs, equation);
                    const int column = trace_offset +
                        ByNodes(column_scalar, scalar_trace_dofs, variable);
                    const double derivative = physical_boundary
                        ? (variable == equation ? 1.0 : 0.0)
                        : trace_jacobian[equation][variable] -
                            (variable == equation ? stabilization : 0.0);
                    system.d[element_d +
                        static_cast<std::size_t>(row) * local_trace +
                        column] += weight *
                        face_trace_shape_workspace[row_scalar] *
                        face_trace_shape_workspace[column_scalar] *
                        derivative;
                  }

            if (regularized)
            {
              for (int row_scalar = 0;
                   row_scalar < scalar_interior_dofs; ++row_scalar)
              {
                for (int direction = 0;
                     direction < cpu_dimension; ++direction)
                {
                  const int row = ByNodes(
                      row_scalar, scalar_interior_dofs, 4 + direction);
                  system.residual_interior[element_ni + row] -= weight *
                      face_interior_shape_workspace[row_scalar] *
                      normal[direction] * sigma_hat;
                  for (int column_scalar = 0;
                       column_scalar < scalar_trace_dofs; ++column_scalar)
                  {
                    const int column = trace_offset +
                        ByNodes(column_scalar, scalar_trace_dofs, 4);
                    system.b[element_b +
                        static_cast<std::size_t>(row) * local_trace +
                        column] -= weight *
                        face_interior_shape_workspace[row_scalar] *
                        normal[direction] *
                        face_trace_shape_workspace[column_scalar];
                  }
                }

                const int sigma_row =
                    ByNodes(row_scalar, scalar_interior_dofs, 6);
                system.residual_interior[element_ni + sigma_row] -=
                    alpha * weight *
                    face_interior_shape_workspace[row_scalar] *
                    z_hat_normal;
                for (int column_scalar = 0;
                     column_scalar < scalar_interior_dofs; ++column_scalar)
                {
                  const double column_shape =
                      face_interior_shape_workspace[column_scalar];
                  system.a[element_a +
                      static_cast<std::size_t>(sigma_row) * local_interior +
                      ByNodes(column_scalar, scalar_interior_dofs, 4)] +=
                      alpha * weight *
                      face_interior_shape_workspace[row_scalar] * tau *
                      column_shape;
                  for (int direction = 0;
                       direction < cpu_dimension; ++direction)
                    system.a[element_a +
                        static_cast<std::size_t>(sigma_row) * local_interior +
                        ByNodes(column_scalar, scalar_interior_dofs,
                                5 + direction)] -=
                        alpha * weight *
                        face_interior_shape_workspace[row_scalar] *
                        normal[direction] * column_shape;
                }
                for (int column_scalar = 0;
                     column_scalar < scalar_trace_dofs; ++column_scalar)
                {
                  const double column_shape =
                      face_trace_shape_workspace[column_scalar];
                  system.b[element_b +
                      static_cast<std::size_t>(sigma_row) * local_trace +
                      trace_offset +
                      ByNodes(column_scalar, scalar_trace_dofs, 4)] -=
                      alpha * weight *
                      face_interior_shape_workspace[row_scalar] * tau *
                      column_shape;
                  system.b[element_b +
                      static_cast<std::size_t>(sigma_row) * local_trace +
                      trace_offset +
                      ByNodes(column_scalar, scalar_trace_dofs, 0)] -=
                      alpha * weight *
                      face_interior_shape_workspace[row_scalar] * tau *
                      jump / trace_state[0] * column_shape;
                }
              }

              for (int row_scalar = 0;
                   row_scalar < scalar_trace_dofs; ++row_scalar)
              {
                const int row = trace_offset +
                    ByNodes(row_scalar, scalar_trace_dofs, 4);
                system.residual_trace[element_nt + row] +=
                    alpha * weight *
                    face_trace_shape_workspace[row_scalar] *
                    z_hat_normal;
                for (int column_scalar = 0;
                     column_scalar < scalar_interior_dofs; ++column_scalar)
                {
                  const double column_shape =
                      face_interior_shape_workspace[column_scalar];
                  system.c[element_c +
                      static_cast<std::size_t>(row) * local_interior +
                      ByNodes(column_scalar, scalar_interior_dofs, 4)] -=
                      alpha * weight *
                      face_trace_shape_workspace[row_scalar] * tau *
                      column_shape;
                  for (int direction = 0;
                       direction < cpu_dimension; ++direction)
                    system.c[element_c +
                        static_cast<std::size_t>(row) * local_interior +
                        ByNodes(column_scalar, scalar_interior_dofs,
                                5 + direction)] +=
                        alpha * weight *
                        face_trace_shape_workspace[row_scalar] *
                        normal[direction] * column_shape;
                }
                for (int column_scalar = 0;
                     column_scalar < scalar_trace_dofs; ++column_scalar)
                {
                  const double column_shape =
                      face_trace_shape_workspace[column_scalar];
                  system.d[element_d +
                      static_cast<std::size_t>(row) * local_trace +
                      trace_offset +
                      ByNodes(column_scalar, scalar_trace_dofs, 4)] +=
                      alpha * weight *
                      face_trace_shape_workspace[row_scalar] * tau *
                      column_shape;
                  system.d[element_d +
                      static_cast<std::size_t>(row) * local_trace +
                      trace_offset +
                      ByNodes(column_scalar, scalar_trace_dofs, 0)] +=
                      alpha * weight *
                      face_trace_shape_workspace[row_scalar] * tau *
                      jump / trace_state[0] * column_shape;
                }
              }
            }

            if (physical_boundary)
            {
              bool c_is_zero = true;
              bool d_is_nonzero = false;
              for (int row_scalar = 0;
                   row_scalar < scalar_trace_dofs; ++row_scalar)
                for (int component = 0;
                     component < cpu_conserved_components; ++component)
                {
                  const int row = trace_offset +
                      ByNodes(row_scalar, scalar_trace_dofs, component);
                  for (int column = 0; column < local_interior; ++column)
                    c_is_zero = c_is_zero &&
                        system.c[element_c +
                                 static_cast<std::size_t>(row) *
                                     local_interior +
                                 column] == 0.0;
                  d_is_nonzero = d_is_nonzero ||
                      system.d[element_d +
                               static_cast<std::size_t>(row) *
                                   local_trace +
                               row] != 0.0;
                }
              if (consume_internal_test_hooks && c_is_zero)
                ++generic_boundary_picard_test_state.zero_c_rows;
              if (consume_internal_test_hooks && d_is_nonzero)
                ++generic_boundary_picard_test_state.nonzero_d_rows;
            }
          }
          trace_offset += scalar_trace_dofs * trace_components;
        }
        if (trace_offset != local_trace) { return false; }
      }

      if (frozen_borrowed_llf &&
          borrowed_llf_index != frozen_borrowed_llf->size())
      {
        return false;
      }

      std::transform(system.residual_interior.begin(),
                     system.residual_interior.end(), system.f.begin(),
                     [](const double value) { return -value; });
      std::transform(system.residual_trace.begin(),
                     system.residual_trace.end(), system.g.begin(),
                     [](const double value) { return -value; });
      return Finite(system.residual_interior) &&
          Finite(system.residual_trace) && Finite(system.a) &&
          Finite(system.b) && Finite(system.c) && Finite(system.d);
    }
    catch (...)
    {
      return false;
    }
  }


  double ResidualNorm(const LocalSystem& system,
                      const bool consume_internal_test_hooks = true) noexcept
  {
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    double local_interior_squared = 0.0;
    bool local_prepared = true;
    try
    {
      local_prepared = system.elements == mesh->GetNE() &&
          system.trace_dofs == local_trace &&
          signed_trace_dofs.size() ==
              static_cast<std::size_t>(system.elements * system.trace_dofs) &&
          system.residual_trace.size() == signed_trace_dofs.size();
      local_trace_residual = 0.0;
      for (int element = 0; element < system.elements && local_prepared;
           ++element)
        for (int local = 0; local < system.trace_dofs; ++local)
        {
          double sign = 1.0;
          const int dof = Decode(signed_trace_dofs[
              static_cast<std::size_t>(element) * system.trace_dofs + local],
                                 sign);
          if (dof < 0 || dof >= local_trace_residual.Size())
          {
            local_prepared = false;
            break;
          }
          local_trace_residual[dof] += sign * system.residual_trace[
              static_cast<std::size_t>(element) * system.trace_dofs + local];
        }
      for (const double value : system.residual_interior)
      {
        local_interior_squared += value * value;
      }
      local_prepared = local_prepared && Finite(local_trace_residual) &&
          Finite(local_interior_squared) &&
          trace_space && trace_space->Dof_TrueDof_Matrix() &&
          (!consume_internal_test_hooks ||
           !ConsumeOwnedCollectiveTestFault(
               detail::CpuCollectiveTestFault::residual_map_preparation, rank,
               incarnation));
    }
    catch (...)
    {
      local_prepared = false;
    }
    if (!CollectiveAll(local_prepared)) { return invalid; }

    bool projection_ready = true;
    try
    {
      trace_space->Dof_TrueDof_Matrix()->MultTranspose(
          local_trace_residual, true_trace_residual);
      projection_ready = Finite(true_trace_residual);
    }
    catch (...)
    {
      projection_ready = false;
    }
    if (!CollectiveAll(projection_ready)) { return invalid; }

    double global_interior_squared = 0.0;
    const int interior_status = MPI_Allreduce(
        &local_interior_squared, &global_interior_squared, 1, MPI_DOUBLE,
        MPI_SUM, communicator);
    double trace_squared = invalid;
    bool trace_reduced = true;
    try
    {
      trace_squared = mfem::InnerProduct(
          communicator, true_trace_residual, true_trace_residual);
    }
    catch (...)
    {
      trace_reduced = false;
    }
    const bool reduced = interior_status == MPI_SUCCESS && trace_reduced &&
        Finite(global_interior_squared) && Finite(trace_squared) &&
        global_interior_squared >= 0.0 && trace_squared >= 0.0;
    if (!CollectiveAll(reduced)) { return invalid; }
    return std::sqrt(global_interior_squared + trace_squared);
  }

  bool CheckState(const State& state, double& minimum_density,
                  double& minimum_pressure,
                  double& maximum_wave_speed) const noexcept
  {
    State primitive{};
    if (!Primitive(state, primitive)) { return false; }
    const double wave_speed =
        std::sqrt(primitive[1] * primitive[1] +
                  primitive[2] * primitive[2]) +
        std::sqrt(cpu_gamma * primitive[3] / primitive[0]);
    if (!Finite(wave_speed)) { return false; }
    minimum_density = std::min(minimum_density, primitive[0]);
    minimum_pressure = std::min(minimum_pressure, primitive[3]);
    maximum_wave_speed = std::max(maximum_wave_speed, wave_speed);
    return IsBorrowed() ||
        (primitive[0] >= density_gate && primitive[3] >= pressure_gate &&
         wave_speed < cpu_fixed_tau);
  }

  bool Prescreen(const mfem::ParGridFunction& interior,
                 const mfem::ParGridFunction& trace_values,
                 double& minimum_density, double& minimum_pressure,
                 double& maximum_wave_speed) noexcept
  {
    double local_density = std::numeric_limits<double>::infinity();
    double local_pressure = std::numeric_limits<double>::infinity();
    double local_wave = 0.0;
    bool local_ok = true;
    try
    {
      for (int element = 0; element < mesh->GetNE() && local_ok; ++element)
      {
        const mfem::FiniteElement* finite_element =
            interior_space->GetFE(element);
        const int* interior_vdofs = element_interior_vdofs.data() +
            static_cast<std::size_t>(element) * local_interior;
        const mfem::IntegrationRule& rule = mfem::IntRules.Get(
            finite_element->GetGeomType(), config.assembly_quadrature_order);
        for (int q = 0; q < rule.GetNPoints() && local_ok; ++q)
        {
          finite_element->CalcShape(rule.IntPoint(q),
                                    volume_shape_workspace);
          State state{};
          local_ok = Interpolate(interior, interior_vdofs,
                                 scalar_interior_dofs,
                                 volume_shape_workspace, state) &&
              CheckState(state, local_density, local_pressure, local_wave);
        }
        for (int local_face = 0; local_face < 4 && local_ok; ++local_face)
        {
          const int face = element_faces[
              static_cast<std::size_t>(element) * 4 + local_face];
          mfem::FaceElementTransformations* transformations =
              mesh->GetFaceElementTransformations(face);
          const mfem::FiniteElement* trace_element =
              trace_space->GetFaceElement(face);
          const bool element_one = transformations->Elem1No == element;
          const int* trace_vdofs = signed_trace_dofs.data() +
              static_cast<std::size_t>(element) * local_trace +
              local_face * scalar_trace_dofs * trace_components;
          const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
              trace_element->GetGeomType(), config.assembly_quadrature_order);
          for (int q = 0; q < face_rule.GetNPoints() && local_ok; ++q)
          {
            const mfem::IntegrationPoint& point = face_rule.IntPoint(q);
            transformations->SetAllIntPoints(&point);
            finite_element->CalcShape(
                element_one ? transformations->GetElement1IntPoint()
                            : transformations->GetElement2IntPoint(),
                face_interior_shape_workspace);
            trace_element->CalcShape(point, face_trace_shape_workspace);
            State side_state{}, trace_state{};
            local_ok = Interpolate(interior, interior_vdofs,
                                   scalar_interior_dofs,
                                   face_interior_shape_workspace,
                                   side_state) &&
                CheckState(side_state, local_density, local_pressure,
                           local_wave) &&
                Interpolate(trace_values, trace_vdofs, scalar_trace_dofs,
                            face_trace_shape_workspace, trace_state) &&
                CheckState(trace_state, local_density, local_pressure,
                           local_wave);
          }
        }
      }
    }
    catch (...)
    {
      local_ok = false;
    }
    int valid = local_ok ? 1 : 0;
    int global_valid = 0;
    const int valid_status = MPI_Allreduce(
        &valid, &global_valid, 1, MPI_INT, MPI_MIN, communicator);
    const int density_status = MPI_Allreduce(
        &local_density, &minimum_density, 1, MPI_DOUBLE, MPI_MIN,
        communicator);
    const int pressure_status = MPI_Allreduce(
        &local_pressure, &minimum_pressure, 1, MPI_DOUBLE, MPI_MIN,
        communicator);
    const int wave_status = MPI_Allreduce(
        &local_wave, &maximum_wave_speed, 1, MPI_DOUBLE, MPI_MAX,
        communicator);
    return valid_status == MPI_SUCCESS && density_status == MPI_SUCCESS &&
        pressure_status == MPI_SUCCESS && wave_status == MPI_SUCCESS &&
        global_valid == 1 && Finite(minimum_density) &&
        Finite(minimum_pressure) && Finite(maximum_wave_speed) &&
        (IsBorrowed()
             ? minimum_density > 0.0 && minimum_pressure > 0.0
             : minimum_density >= density_gate &&
                   minimum_pressure >= pressure_gate &&
                   maximum_wave_speed < cpu_fixed_tau);
  }

  bool ConservationBalance(
      const mfem::ParGridFunction& current_interior,
      const mfem::ParGridFunction& current_trace,
      const double dt,
      double& maximum_error,
      double& maximum_relative_error) noexcept
  {
    maximum_error = 0.0;
    maximum_relative_error = 0.0;
    if (!IsBorrowed()) { return true; }
    std::array<double, 5> local_balance{};
    std::array<double, 5> local_scale{};
    bool local_ok = Finite(dt) && dt > 0.0 && external_source != nullptr;
    try
    {
      const bool regularized = IsIgr();
      const double alpha = regularized
          ? borrowed_request->numerical_policy.stabilization.alpha : 0.0;
      const double beta = regularized
          ? borrowed_request->numerical_policy.stabilization.beta : 0.0;
      const double degree_scale =
          static_cast<double>((config.polynomial_order + 1) *
                              (config.polynomial_order + 1));
      for (int element = 0; element < mesh->GetNE() && local_ok; ++element)
      {
        const mfem::FiniteElement* finite_element =
            interior_space->GetFE(element);
        mfem::ElementTransformation* transformation =
            mesh->GetElementTransformation(element);
        if (!finite_element || !transformation)
        {
          local_ok = false;
          break;
        }
        const int* interior_vdofs = element_interior_vdofs.data() +
            static_cast<std::size_t>(element) * local_interior;
        const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
            finite_element->GetGeomType(), config.assembly_quadrature_order);
        for (int q = 0; q < volume_rule.GetNPoints() && local_ok; ++q)
        {
          const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
          transformation->SetIntPoint(&point);
          finite_element->CalcShape(point, volume_shape_workspace);
          if (regularized)
            finite_element->CalcPhysDShape(*transformation,
                                           volume_derivative_workspace);
          State current{}, previous{}, source{};
          local_ok = Interpolate(current_interior, interior_vdofs,
                                 scalar_interior_dofs,
                                 volume_shape_workspace, current) &&
              Interpolate(*old_interior, interior_vdofs,
                          scalar_interior_dofs,
                          volume_shape_workspace, previous) &&
              Interpolate(*external_source, interior_vdofs,
                          scalar_interior_dofs,
                          volume_shape_workspace, source);
          double sigma = 0.0;
          std::array<std::array<double, 2>, 3> conservative_gradient{};
          if (local_ok && regularized)
          {
            local_ok = InterpolateComponent(
                current_interior, interior_vdofs, scalar_interior_dofs,
                volume_shape_workspace, 4, sigma);
            for (int component = 0; component < 3 && local_ok; ++component)
              for (int direction = 0;
                   direction < cpu_dimension && local_ok; ++direction)
                local_ok = InterpolateDerivative(
                    current_interior, interior_vdofs, scalar_interior_dofs,
                    volume_derivative_workspace, component, direction,
                    conservative_gradient[component][direction]);
          }
          if (!local_ok) { break; }
          const double weight = point.weight * transformation->Weight();
          if (!Finite(weight) || !(weight > 0.0))
          {
            local_ok = false;
            break;
          }
          for (int component = 0;
               component < cpu_conserved_components; ++component)
          {
            const double change = weight *
                (current[component] - previous[component]);
            const double forcing = dt * weight * source[component];
            local_balance[component] += change - forcing;
            local_scale[component] +=
                std::abs(change) + std::abs(forcing);
          }
          if (regularized)
          {
            if (!(current[0] > density_gate) || !Finite(sigma))
            {
              local_ok = false;
              break;
            }
            std::array<std::array<double, 2>, 2> velocity_gradient{};
            for (int velocity = 0; velocity < cpu_dimension; ++velocity)
              for (int direction = 0;
                   direction < cpu_dimension; ++direction)
                velocity_gradient[velocity][direction] =
                    conservative_gradient[1 + velocity][direction] /
                        current[0] -
                    current[1 + velocity] *
                        conservative_gradient[0][direction] /
                        (current[0] * current[0]);
            const double velocity_trace =
                velocity_gradient[0][0] + velocity_gradient[1][1];
            const double q =
                velocity_gradient[0][0] * velocity_gradient[0][0] +
                2.0 * velocity_gradient[0][1] *
                    velocity_gradient[1][0] +
                velocity_gradient[1][1] * velocity_gradient[1][1] +
                velocity_trace * velocity_trace;
            const double sigma_contribution = weight * sigma / current[0];
            const double q_contribution = weight * alpha * q;
            if (!Finite(q) || !Finite(sigma_contribution) ||
                !Finite(q_contribution))
            {
              local_ok = false;
              break;
            }
            local_balance[4] += sigma_contribution - q_contribution;
            local_scale[4] +=
                std::abs(sigma_contribution) + std::abs(q_contribution);
          }
        }

        for (int local_face = 0;
             local_face < 4 && local_ok; ++local_face)
        {
          const int face = element_faces[
              static_cast<std::size_t>(element) * 4 + local_face];
          mfem::FaceElementTransformations* face_transformation =
              mesh->GetFaceElementTransformations(face);
          const mfem::FiniteElement* trace_element =
              trace_space->GetFaceElement(face);
          const bool true_boundary = face >= 0 &&
              face < static_cast<int>(shared_faces.size()) &&
              shared_faces[static_cast<std::size_t>(face)] == 0 &&
              face_transformation && face_transformation->Elem2No < 0 &&
              mesh->GetFaceToBdrElMap()[face] >= 0;
          if (!true_boundary) { continue; }
          if (!face_transformation || !face_transformation->Face ||
              !trace_element)
          {
            local_ok = false;
            break;
          }
          const bool element_one =
              face_transformation->Elem1No == element;
          if (!element_one && face_transformation->Elem2No != element)
          {
            local_ok = false;
            break;
          }
          const int trace_offset = local_face * scalar_trace_dofs *
              trace_components;
          const int* face_trace_vdofs = signed_trace_dofs.data() +
              static_cast<std::size_t>(element) * local_trace +
              trace_offset;
          const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
              trace_element->GetGeomType(),
              config.assembly_quadrature_order);
          double face_volume = 0.0;
          for (int measure_q = 0;
               measure_q < face_rule.GetNPoints(); ++measure_q)
          {
            const mfem::IntegrationPoint& measure_point =
                face_rule.IntPoint(measure_q);
            face_transformation->SetAllIntPoints(&measure_point);
            face_volume += measure_point.weight *
                face_transformation->Face->Weight();
          }
          const double h = regularized
              ? mesh->GetElementVolume(element) / face_volume : 1.0;
          if (!Finite(face_volume) || !(face_volume > 0.0) ||
              (regularized && (!Finite(h) || !(h > 0.0))))
          {
            local_ok = false;
            break;
          }
          for (int q = 0; q < face_rule.GetNPoints() && local_ok; ++q)
          {
            const mfem::IntegrationPoint& face_point =
                face_rule.IntPoint(q);
            face_transformation->SetAllIntPoints(&face_point);
            const mfem::IntegrationPoint& element_point = element_one
                ? face_transformation->GetElement1IntPoint()
                : face_transformation->GetElement2IntPoint();
            mfem::CalcOrtho(face_transformation->Face->Jacobian(),
                            scaled_normal_workspace);
            const double normal_norm = scaled_normal_workspace.Norml2();
            const double face_measure =
                face_transformation->Face->Weight();
            if (!Finite(normal_norm) || !(normal_norm > 0.0) ||
                !Finite(face_measure) || !(face_measure > 0.0))
            {
              local_ok = false;
              break;
            }
            const double side = element_one ? 1.0 : -1.0;
            const std::array<double, 2> normal{{
                side * scaled_normal_workspace[0] / normal_norm,
                side * scaled_normal_workspace[1] / normal_norm}};
            finite_element->CalcShape(element_point,
                                      face_interior_shape_workspace);
            trace_element->CalcShape(face_point,
                                     face_trace_shape_workspace);
            State interior{}, trace_state{}, flux{};
            Matrix45 ignored{};
            local_ok = Interpolate(current_interior, interior_vdofs,
                                   scalar_interior_dofs,
                                   face_interior_shape_workspace,
                                   interior) &&
                Interpolate(current_trace, face_trace_vdofs,
                            scalar_trace_dofs,
                            face_trace_shape_workspace, trace_state);
            if (!local_ok) { break; }
            double sigma = 0.0;
            double sigma_hat = 0.0;
            std::array<double, 2> z{};
            if (regularized)
            {
              local_ok = InterpolateComponent(
                      current_interior, interior_vdofs,
                      scalar_interior_dofs,
                      face_interior_shape_workspace, 4, sigma) &&
                  InterpolateComponent(
                      current_interior, interior_vdofs,
                      scalar_interior_dofs,
                      face_interior_shape_workspace, 5, z[0]) &&
                  InterpolateComponent(
                      current_interior, interior_vdofs,
                      scalar_interior_dofs,
                      face_interior_shape_workspace, 6, z[1]) &&
                  InterpolateComponent(
                      current_trace, face_trace_vdofs,
                      scalar_trace_dofs,
                      face_trace_shape_workspace, 4, sigma_hat) &&
                  RegularizedNormalFlux(trace_state, sigma_hat, normal,
                                        flux, ignored);
            }
            else
            {
              local_ok = NormalFlux(trace_state, normal, flux);
            }
            if (!local_ok) { break; }
            double stabilization = beta;
            if (!regularized)
            {
              local_ok = BorrowedEulerLlfStabilization(
                  interior, trace_state, normal, stabilization);
            }
            if (!local_ok) { break; }
            const double weight = face_point.weight * face_measure;
            for (int component = 0;
                 component < cpu_conserved_components; ++component)
            {
              const double numerical_flux = flux[component] +
                  stabilization *
                      (interior[component] - trace_state[component]);
              const double contribution = dt * weight * numerical_flux;
              local_balance[component] += contribution;
              local_scale[component] += std::abs(contribution);
            }
            if (regularized)
            {
              const double tau = degree_scale / (trace_state[0] * h);
              const double z_hat_normal =
                  z[0] * normal[0] + z[1] * normal[1] -
                  tau * (sigma - sigma_hat);
              const double contribution =
                  alpha * weight * z_hat_normal;
              if (!Finite(tau) || !(tau > 0.0) ||
                  !Finite(contribution))
              {
                local_ok = false;
                break;
              }
              local_balance[4] -= contribution;
              local_scale[4] += std::abs(contribution);
            }
          }
        }
      }
    }
    catch (...)
    {
      local_ok = false;
    }
    if (!CollectiveAll(local_ok && Finite(local_balance) &&
                       Finite(local_scale)))
    {
      return false;
    }
    std::array<double, 5> global_balance{};
    std::array<double, 5> global_scale{};
    const int balance_status = MPI_Allreduce(
        local_balance.data(), global_balance.data(),
        static_cast<int>(global_balance.size()), MPI_DOUBLE, MPI_SUM,
        communicator);
    const int scale_status = MPI_Allreduce(
        local_scale.data(), global_scale.data(),
        static_cast<int>(global_scale.size()), MPI_DOUBLE, MPI_SUM,
        communicator);
    if (!CollectiveAll(balance_status == MPI_SUCCESS &&
                       scale_status == MPI_SUCCESS &&
                       Finite(global_balance) && Finite(global_scale)))
    {
      return false;
    }
    const int components = IsIgr() ? 5 : cpu_conserved_components;
    for (int component = 0; component < components; ++component)
    {
      const double error = std::abs(global_balance[component]);
      const double relative = error / (1.0 + global_scale[component]);
      maximum_error = std::max(maximum_error, error);
      maximum_relative_error = std::max(maximum_relative_error, relative);
    }
    return Finite(maximum_error) && Finite(maximum_relative_error);
  }

  /**
   * Compute a CFL diagnostic from a borrowed slot.  Both the volume and
   * trace representations are reloaded from the supplied state so the
   * reported wave speed has the same observable scope as publication.
   */
  bool ScanBorrowedEulerState(const CpuSlotStorage& committed_state,
                              double& maximum_wave_speed,
                              double& backend_stable_dt) noexcept
  {
    maximum_wave_speed = 0.0;
    backend_stable_dt = std::numeric_limits<double>::quiet_NaN();
    const bool local_state_ready = initialized && IsBorrowed() && mesh &&
        iterate_interior && iterate_trace;
    if (!CollectiveAll(local_state_ready)) { return false; }

    const bool fields_assigned =
        Assign(*iterate_interior, committed_state.interior) &&
        Assign(iterate_true_trace, committed_state.trace);
    if (!CollectiveAll(fields_assigned)) { return false; }
    // LoadTrace owns parallel trace distribution.  Every rank reaches it
    // only after the rank-local slot copies have collectively succeeded.
    const bool trace_loaded = LoadTrace(iterate_true_trace, *iterate_trace);
    if (!CollectiveAll(trace_loaded)) { return false; }

    double minimum_density = 0.0;
    double minimum_pressure = 0.0;
    const bool state_valid = Prescreen(*iterate_interior, *iterate_trace,
                                       minimum_density, minimum_pressure,
                                       maximum_wave_speed);
    if (!CollectiveAll(state_valid)) { return false; }

    double local_hmin = std::numeric_limits<double>::infinity();
    bool local_geometry_valid = true;
    try
    {
      for (int element = 0; element < mesh->GetNE(); ++element)
      {
        const double element_hmin = mesh->GetElementSize(element, 1);
        if (!Finite(element_hmin) || !(element_hmin > 0.0))
        {
          local_geometry_valid = false;
          break;
        }
        local_hmin = std::min(local_hmin, element_hmin);
      }
    }
    catch (...)
    {
      local_geometry_valid = false;
    }
    double global_hmin = std::numeric_limits<double>::infinity();
    const int hmin_status = MPI_Allreduce(
        &local_hmin, &global_hmin, 1, MPI_DOUBLE, MPI_MIN, communicator);
    const bool geometry_valid = local_geometry_valid &&
        hmin_status == MPI_SUCCESS && Finite(global_hmin) &&
        global_hmin > 0.0;
    if (!CollectiveAll(geometry_valid)) { return false; }

    const double denominator =
        2.0 * static_cast<double>(config.polynomial_order) + 1.0;
    backend_stable_dt = maximum_wave_speed > 0.0
        ? global_hmin / (denominator * maximum_wave_speed)
        : global_hmin / denominator;
    return CollectiveAll(Finite(maximum_wave_speed) &&
                         maximum_wave_speed >= 0.0 &&
                         Finite(backend_stable_dt) && backend_stable_dt > 0.0);
  }

  bool CondenseAndAssemble(const LocalSystem& system,
                           StateHash& matrix_hash,
                           std::uint64_t& matrix_refresh_generation) noexcept
  {
    matrix_refresh_generation = 0;
    double* diagonal_data = nullptr;
    double* offdiagonal_data = nullptr;
    bool outgoing_ready = false;
    try
    {
      const auto prepare_outgoing = [&]() -> bool
      {
        if (system.elements == 0)
        {
          *local_matrix = 0.0;
          local_trace_rhs = 0.0;
          return outgoing_schur_terms.empty() && schur.empty() &&
              reduced_rhs.empty();
        }
        const LocalBlockShape shape{
            local_interior / interior_components,
            local_trace / trace_components,
            interior_components, trace_components};
        const core::LocalSystemBatchView input{
            shape,
            AsConst(View(system.a.data(), system.elements, local_interior,
                         local_interior)),
            AsConst(View(system.b.data(), system.elements, local_interior,
                         local_trace)),
            AsConst(View(system.c.data(), system.elements, local_trace,
                         local_interior)),
            AsConst(View(system.d.data(), system.elements, local_trace,
                         local_trace)),
            AsConst(View(system.f.data(), system.elements, local_interior, 1)),
            AsConst(View(system.g.data(), system.elements, local_trace, 1))};
        if (condensation.Condense(
                input,
                {View(schur.data(), system.elements, local_trace, local_trace),
                 View(reduced_rhs.data(), system.elements, local_trace, 1)}) !=
                core::Status::success ||
            !Finite(schur) || !Finite(reduced_rhs))
        {
          return false;
        }

        *local_matrix = 0.0;
        local_trace_rhs = 0.0;
        for (int element = 0; element < system.elements; ++element)
        {
          for (int row = 0; row < local_trace; ++row)
          {
            const int signed_dof = signed_trace_dofs[
                static_cast<std::size_t>(element) * local_trace + row];
            element_trace_dofs_workspace[row] = signed_dof;
            double sign = 1.0;
            const int dof = Decode(signed_dof, sign);
            if (dof < 0 || dof >= local_trace_rhs.Size()) { return false; }
            local_trace_rhs[dof] += sign * reduced_rhs[
                static_cast<std::size_t>(element) * local_trace + row];
            for (int column = 0; column < local_trace; ++column)
            {
              local_schur_workspace(row, column) = schur[
                  static_cast<std::size_t>(element) * local_trace *
                      local_trace +
                  static_cast<std::size_t>(row) * local_trace + column];
            }
          }
          local_matrix->AddSubMatrix(element_trace_dofs_workspace,
                                     element_trace_dofs_workspace,
                                     local_schur_workspace, 1);
        }
        if (!true_matrix) { return false; }
        true_matrix->HostReadWrite();
        true_matrix->GetDiag(true_diagonal_view);
        true_matrix->GetOffd(true_offdiagonal_view,
                             true_offdiagonal_columns);
        diagonal_data = true_diagonal_view.GetData();
        offdiagonal_data = true_offdiagonal_view.GetData();
        if (!diagonal_data ||
            (true_offdiagonal_view.NumNonZeroElems() != 0 &&
             !offdiagonal_data))
        {
          return false;
        }
        std::fill(diagonal_data,
                  diagonal_data + true_diagonal_view.NumNonZeroElems(), 0.0);
        if (offdiagonal_data)
        {
          std::fill(offdiagonal_data,
                    offdiagonal_data + true_offdiagonal_view.NumNonZeroElems(),
                    0.0);
        }
        for (const SparseSchurTerm& term : owned_schur_terms)
        {
          if (term.schur_index >= schur.size() || term.data_index < 0)
          {
            return false;
          }
          double* destination = term.offdiagonal ? offdiagonal_data
                                                 : diagonal_data;
          if (!destination) { return false; }
          destination[term.data_index] += term.sign * schur[term.schur_index];
        }
        for (std::size_t index = 0; index < outgoing_schur_terms.size();
             ++index)
        {
          const SparseOutgoingTerm& term = outgoing_schur_terms[index];
          if (term.schur_index >= schur.size()) { return false; }
          outgoing_schur_values[index] = term.sign * schur[term.schur_index];
        }
        return outgoing_schur_values.size() <=
                   static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
            incoming_schur_values.size() <=
                   static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
            Finite(outgoing_schur_values) &&
            !ConsumeOwnedCollectiveTestFault(
                detail::CpuCollectiveTestFault::
                    condensation_outgoing_preparation,
                rank, incarnation);
      };
      outgoing_ready = prepare_outgoing();
    }
    catch (...)
    {
      outgoing_ready = false;
    }
    if (!CollectiveAll(outgoing_ready)) { return false; }

    const bool exchange_ready = MPI_Alltoallv(
        outgoing_schur_values.data(), outgoing_schur_counts.data(),
        outgoing_schur_offsets.data(), MPI_DOUBLE, incoming_schur_values.data(),
        incoming_schur_counts.data(), incoming_schur_offsets.data(), MPI_DOUBLE,
        communicator) == MPI_SUCCESS;
    if (!CollectiveAll(exchange_ready)) { return false; }

    bool incoming_ready = false;
    try
    {
      const auto apply_incoming = [&]() -> bool
      {
        if (!Finite(incoming_schur_values)) { return false; }
        for (std::size_t index = 0; index < incoming_schur_slots.size();
             ++index)
        {
          const SparseIncomingSlot& slot = incoming_schur_slots[index];
          if (slot.data_index < 0) { return false; }
          double* destination = slot.offdiagonal ? offdiagonal_data
                                                 : diagonal_data;
          if (!destination) { return false; }
          destination[slot.data_index] += incoming_schur_values[index];
        }
        return true;
      };
      incoming_ready = apply_incoming();
    }
    catch (...)
    {
      incoming_ready = false;
    }
    if (!CollectiveAll(incoming_ready)) { return false; }

    bool solver_binding_ready = true;
    try
    {
      true_matrix->HypreRead();
      solver_binding_ready = RebindTraceSolverAfterMatrixRefresh();
    }
    catch (...)
    {
      solver_binding_ready = false;
    }
    if (!CollectiveAll(solver_binding_ready)) { return false; }
    pending_factor_setup_generation = next_matrix_refresh_generation++;
    matrix_refresh_generation = pending_factor_setup_generation;

    bool true_assembly_ready = true;
    try
    {
      trace_space->Dof_TrueDof_Matrix()->MultTranspose(
          local_trace_rhs, true_trace_rhs);
      true_assembly_ready = Finite(true_trace_rhs);
    }
    catch (...)
    {
      true_assembly_ready = false;
    }
    if (!CollectiveAll(true_assembly_ready)) { return false; }
    matrix_hash = HashPair(HashSystemMatrix(system), HashDoubles(schur));
    return true;
  }

  bool SolveTrace(const std::uint64_t expected_matrix_refresh_generation,
                  double& true_relative_residual, int& iterations,
                  StateHash& factor_hash,
                  StateHash& solution_hash,
                  std::uint64_t& factor_setup_generation,
                  bool& factor_setup_current) noexcept
  {
    factor_setup_generation = 0;
    factor_setup_current = false;
    bool rhs_ready = true;
    try
    {
      if (expected_matrix_refresh_generation == 0 ||
          pending_factor_setup_generation !=
              expected_matrix_refresh_generation ||
          complete_factor_setup_generation >=
              expected_matrix_refresh_generation)
      {
        rhs_ready = false;
      }
      const double* rhs_source = true_trace_rhs.HostRead();
      double* rhs_destination = hypre_rhs->HostWrite();
      if ((!rhs_source || !rhs_destination) && true_trace_rhs.Size() != 0)
      {
        rhs_ready = false;
      }
      if (rhs_ready)
      {
        std::memcpy(rhs_destination, rhs_source,
                    static_cast<std::size_t>(true_trace_rhs.Size()) *
                        sizeof(double));
        *hypre_solution = 0.0;
      }
    }
    catch (...)
    {
      rhs_ready = false;
    }
    if (!CollectiveAll(rhs_ready)) { return false; }

    bool solved = true;
    try
    {
      gmres->Mult(*hypre_rhs, *hypre_solution);
      factor_setup_generation = pending_factor_setup_generation;
      factor_setup_current = factor_setup_generation ==
          expected_matrix_refresh_generation;
      gmres->GetNumIterations(iterations);
      const int maximum_iterations =
          generic::ResolveEulerHdgMaximumLinearIterations(
              config.maximum_linear_iterations,
              static_cast<int>(trace_space->GlobalTrueVSize()));
      // A coherent zero right-hand side is already solved by the zero initial
      // guess and HYPRE reports zero iterations.  The rebuilt true residual
      // below remains the acceptance authority for both zero and nonzero
      // iteration counts.
      solved = factor_setup_current && iterations >= 0 &&
          iterations <= maximum_iterations;
    }
    catch (...)
    {
      solved = false;
    }
    if (!CollectiveAll(solved)) { return false; }
    complete_factor_setup_generation = factor_setup_generation;
    pending_factor_setup_generation = 0;

    // This attests the exact Schur input and frozen ILU configuration only;
    // HYPRE does not expose stable factor bytes for hashing.
    constexpr std::string_view factor_input_config{
        "hypre-ilu:t0:f0:i1:tol0:r0:p0"};
    factor_hash = HashPair(
        HashDoubles(schur),
        HashBytes(factor_input_config.data(), factor_input_config.size()));

    bool residual_ready = true;
    try
    {
      true_matrix->Mult(*hypre_solution, *hypre_residual);
      *hypre_residual -= *hypre_rhs;
      residual_ready = Finite(*hypre_residual) && Finite(*hypre_rhs);
    }
    catch (...)
    {
      residual_ready = false;
    }
    if (!CollectiveAll(residual_ready)) { return false; }

    double residual_squared = std::numeric_limits<double>::quiet_NaN();
    double rhs_squared = std::numeric_limits<double>::quiet_NaN();
    bool residual_reduced = true;
    bool rhs_reduced = true;
    try
    {
      residual_squared = mfem::InnerProduct(
          communicator, *hypre_residual, *hypre_residual);
    }
    catch (...)
    {
      residual_reduced = false;
    }
    try
    {
      rhs_squared = mfem::InnerProduct(
          communicator, *hypre_rhs, *hypre_rhs);
    }
    catch (...)
    {
      rhs_reduced = false;
    }
    bool solution_ready = residual_reduced && rhs_reduced &&
        Finite(residual_squared) && Finite(rhs_squared) &&
        residual_squared >= 0.0 && rhs_squared >= 0.0;
    try
    {
      true_relative_residual = std::sqrt(residual_squared) /
          std::max(std::sqrt(rhs_squared), 1.0e-300);
      const double* solution = hypre_solution->HostRead();
      double* increment = true_increment.HostWrite();
      if ((!solution || !increment) && true_increment.Size() != 0)
      {
        solution_ready = false;
      }
      if (solution_ready)
      {
        std::memcpy(increment, solution,
                    static_cast<std::size_t>(true_increment.Size()) *
                        sizeof(double));
        solution_hash = HashBytes(
            solution, static_cast<std::size_t>(true_increment.Size()) *
                          sizeof(double));
      }
      solution_ready = solution_ready && Finite(true_relative_residual) &&
          true_relative_residual <= config.true_residual_tolerance;
    }
    catch (...)
    {
      solution_ready = false;
    }
    return CollectiveAll(solution_ready);
  }

  bool Recover() noexcept
  {
    bool distribution_prepared = false;
    try
    {
      distributed_increment = 0.0;
      distribution_prepared = exchange_plan.has_value();
    }
    catch (...)
    {
      distribution_prepared = false;
    }
    if (!CollectiveAll(distribution_prepared)) { return false; }

    bool distributed_ready = false;
    try
    {
      distributed_ready = exchange_plan->Distribute(
          *trace_space, true_increment, distributed_increment);
    }
    catch (...)
    {
      distributed_ready = false;
    }
    if (!CollectiveAll(distributed_ready)) { return false; }

    bool local_recovery_ready = true;
    try
    {
      const double* distributed = distributed_increment.HostRead();
      if (!distributed && distributed_increment.Size() != 0)
      {
        local_recovery_ready = false;
      }
      for (int element = 0;
           element < mesh->GetNE() && local_recovery_ready; ++element)
        for (int local = 0; local < local_trace; ++local)
        {
          double sign = 1.0;
          const int dof = Decode(signed_trace_dofs[
              static_cast<std::size_t>(element) * local_trace + local], sign);
          if (dof < 0 || dof >= distributed_increment.Size())
          {
            local_recovery_ready = false;
            break;
          }
          local_trace_increment[
              static_cast<std::size_t>(element) * local_trace + local] =
              sign * distributed[dof];
        }
      if (local_recovery_ready && mesh->GetNE() > 0 &&
          (condensation.Recover(
              AsConst(View(local_trace_increment.data(), mesh->GetNE(),
                           local_trace, 1)),
              View(recovered_interior.data(), mesh->GetNE(), local_interior,
                   1)) != core::Status::success ||
           !Finite(recovered_interior)))
      {
        local_recovery_ready = false;
      }
      std::fill(grid_interior_increment.begin(),
                grid_interior_increment.end(), 0.0);
      std::fill(interior_census.begin(), interior_census.end(), 0);
      for (int element = 0;
           element < mesh->GetNE() && local_recovery_ready; ++element)
        for (int local = 0; local < local_interior; ++local)
        {
          double sign = 1.0;
          const int dof = Decode(element_interior_vdofs[
              static_cast<std::size_t>(element) * local_interior + local],
                                 sign);
          if (dof < 0 ||
              dof >= static_cast<int>(grid_interior_increment.size()))
          {
            local_recovery_ready = false;
            break;
          }
          grid_interior_increment[static_cast<std::size_t>(dof)] = sign *
              recovered_interior[
                  static_cast<std::size_t>(element) * local_interior + local];
          ++interior_census[static_cast<std::size_t>(dof)];
        }
      local_recovery_ready = local_recovery_ready &&
          std::all_of(interior_census.begin(), interior_census.end(),
                      [](const int count) { return count == 1; }) &&
          Finite(grid_interior_increment);
    }
    catch (...)
    {
      local_recovery_ready = false;
    }
    return CollectiveAll(local_recovery_ready);
  }

  bool MakeTrial(const double alpha) noexcept
  {
    bool local_trial_ready = true;
    try
    {
      const double* interior = iterate_interior->HostRead();
      double* trial = trial_interior->HostWrite();
      const double* trace_values = iterate_true_trace.HostRead();
      const double* trace_delta = true_increment.HostRead();
      double* trace_trial = trial_true_trace.HostWrite();
      if ((!interior && iterate_interior->Size() != 0) ||
          (!trial && trial_interior->Size() != 0) ||
          (!trace_values && iterate_true_trace.Size() != 0) ||
          (!trace_delta && true_increment.Size() != 0) ||
          (!trace_trial && trial_true_trace.Size() != 0))
      {
        local_trial_ready = false;
      }
      for (int index = 0;
           index < iterate_interior->Size() && local_trial_ready; ++index)
      {
        trial[index] = interior[index] + alpha *
            grid_interior_increment[static_cast<std::size_t>(index)];
      }
      for (int index = 0;
           index < iterate_true_trace.Size() && local_trial_ready; ++index)
      {
        trace_trial[index] = trace_values[index] + alpha * trace_delta[index];
      }
    }
    catch (...)
    {
      local_trial_ready = false;
    }
    if (!CollectiveAll(local_trial_ready)) { return false; }
    return LoadTrace(trial_true_trace, *trial_trace);
  }

  bool AcceptTrial() noexcept
  {
    bool local_accept_ready = true;
    try
    {
      const double* interior = trial_interior->HostRead();
      double* next_interior = iterate_interior->HostWrite();
      const double* trace_values = trial_true_trace.HostRead();
      double* next_trace = iterate_true_trace.HostWrite();
      if ((!interior && trial_interior->Size() != 0) ||
          (!next_interior && iterate_interior->Size() != 0) ||
          (!trace_values && trial_true_trace.Size() != 0) ||
          (!next_trace && iterate_true_trace.Size() != 0))
      {
        local_accept_ready = false;
      }
      if (local_accept_ready)
      {
        if (iterate_interior->Size() != 0)
        {
          std::memcpy(next_interior, interior,
                      static_cast<std::size_t>(iterate_interior->Size()) *
                          sizeof(double));
        }
        if (iterate_true_trace.Size() != 0)
        {
          std::memcpy(next_trace, trace_values,
                      static_cast<std::size_t>(iterate_true_trace.Size()) *
                          sizeof(double));
        }
      }
    }
    catch (...)
    {
      local_accept_ready = false;
    }
    if (!CollectiveAll(local_accept_ready)) { return false; }
    if (!LoadTrace(iterate_true_trace, *iterate_trace)) { return false; }
    bool copied = true;
    try
    {
      CopySystem(trial_system, current_system);
    }
    catch (...)
    {
      copied = false;
    }
    return CollectiveAll(copied);
  }

  bool StoreCandidate() noexcept
  {
    CpuSlotStorage* candidate = Candidate();
    if (!candidate ||
        candidate->interior.size() !=
            static_cast<std::size_t>(iterate_interior->Size()) ||
        candidate->trace.size() !=
            static_cast<std::size_t>(iterate_true_trace.Size()))
    {
      return false;
    }
    return Copy(*iterate_interior, candidate->interior) &&
        Copy(iterate_true_trace, candidate->trace);
  }

};

namespace
{

StateHash CollectiveHash(const StateHash& local, const int ranks,
                         const MPI_Comm communicator = MPI_COMM_WORLD) noexcept
{
  std::vector<StateHash> hashes;
  try { hashes.resize(static_cast<std::size_t>(ranks)); }
  catch (...) { return {}; }
  if (MPI_Allgather(local.bytes.data(), static_cast<int>(local.bytes.size()),
                    MPI_BYTE, hashes.data(),
                    static_cast<int>(local.bytes.size()), MPI_BYTE,
                    communicator) != MPI_SUCCESS)
  {
    return {};
  }
  return HashBytes(hashes.data(),
                   static_cast<std::size_t>(ranks) * sizeof(StateHash));
}

bool AssignGridFunction(mfem::ParGridFunction& destination,
                        const std::vector<double>& source) noexcept
{
  return Assign(destination, source);
}

} // namespace

detail::CpuNumericsCore::CpuNumericsCore()
    : implementation_(std::make_unique<Implementation>()),
      incarnation_(NextCoreIncarnation())
{
  implementation_->incarnation = incarnation_;
  (void)implementation_->allocations.Observe(
      CpuOwnedAllocationSite::core_wrapper, sizeof(Implementation),
      alignof(Implementation), 1, implementation_.get());
}

detail::CpuNumericsCore::~CpuNumericsCore() = default;
detail::CpuNumericsCore::CpuNumericsCore(CpuNumericsCore&&) noexcept = default;
detail::CpuNumericsCore& detail::CpuNumericsCore::operator=(
    CpuNumericsCore&&) noexcept = default;

bool detail::CpuNumericsCore::ResetAfterInitializationFailure() noexcept
{
  // The exchange plan is optional, so an empty Implementation performs no
  // selected W8-owned allocation.  Reverse member destruction releases every
  // dependent solver/space/mesh object before the same fixed core storage is
  // reconstructed.  No make_unique/vector resize occurs on this reset path.
  Implementation* const storage = implementation_.get();
  bool local_ready = storage != nullptr && incarnation_ != 0;
  const int previous_rank = local_ready ? storage->rank : -1;
  if (local_ready)
  {
    storage->~Implementation();
    ::new (static_cast<void*>(storage)) Implementation();
    storage->incarnation = incarnation_;
    local_ready = storage->allocations.Observe(
        CpuOwnedAllocationSite::core_wrapper, sizeof(Implementation),
        alignof(Implementation), 1, storage) && !storage->initialized &&
        !storage->mesh && !storage->interior_space &&
        !storage->trace_space && !storage->exchange_plan &&
        !storage->committed_slot.interior.cell &&
        !storage->candidate_slot.interior.cell && !storage->true_matrix &&
        !storage->gmres && !storage->ilu &&
        !RefuseCollectiveTestReset(incarnation_, previous_rank);
  }
  const bool reset_complete = AllRanks(local_ready);
  RecordCollectiveTestReset(incarnation_, reset_complete);
  return reset_complete;
}

bool detail::CpuNumericsCore::AdmitInitialization(
    const bool facade_empty, const CpuPersistentConfig& config,
    const RuntimeTupleDescriptor& tuple) const noexcept
{
  const Implementation& state = *implementation_;
  int mpi_initialized = 0;
  int mpi_finalized = 0;
  int ranks = 0;
  MPI_Initialized(&mpi_initialized);
  MPI_Finalized(&mpi_finalized);
  const bool config_consensus = mpi_initialized != 0 && mpi_finalized == 0
      ? ExactConfigConsensus(config) : false;
  const bool local_ready = facade_empty && incarnation_ != 0 &&
      state.incarnation == incarnation_ && !state.initialized &&
      mpi_initialized != 0 && mpi_finalized == 0 &&
      MPI_Comm_size(MPI_COMM_WORLD, &ranks) == MPI_SUCCESS &&
      (ranks == 1 || ranks == 2) && ExactConfig(config) &&
      ExactRankShape(config, ranks) && config_consensus &&
      ExactTuple(tuple, ranks);
  return AllRanks(local_ready);
}

bool detail::CpuNumericsCore::CollectiveAll(const bool local) const noexcept
{
  return implementation_ && implementation_->CollectiveAll(local);
}

bool detail::CpuNumericsCore::AdmitBorrowedEulerOperation(
    const CpuBorrowedEulerOperation operation) const noexcept
{
  const Implementation& state = *implementation_;
  const int local_operation = static_cast<int>(operation);
  int minimum_operation = 0;
  int maximum_operation = 0;
  const int minimum_status = MPI_Allreduce(
      &local_operation, &minimum_operation, 1, MPI_INT, MPI_MIN,
      state.communicator);
  const int maximum_status = MPI_Allreduce(
      &local_operation, &maximum_operation, 1, MPI_INT, MPI_MAX,
      state.communicator);
  const bool known = local_operation >=
          static_cast<int>(CpuBorrowedEulerOperation::prepare) &&
      local_operation <= static_cast<int>(
          CpuBorrowedEulerOperation::export_prepared_grid_function);
  return state.CollectiveAll(
      state.initialized && state.IsBorrowed() &&
      minimum_status == MPI_SUCCESS && maximum_status == MPI_SUCCESS && known &&
      minimum_operation == maximum_operation);
}

bool detail::CpuNumericsCore::ConsumeCollectiveTestFault(
    const detail::CpuCollectiveTestFault fault) noexcept
{
  const Implementation& state = *implementation_;
  return ConsumeOwnedCollectiveTestFault(
      fault, state.rank, incarnation_);
}

std::uint64_t detail::CpuNumericsCore::incarnation() const noexcept
{
  return incarnation_;
}

bool detail::CpuNumericsCore::Initialize(
    const CpuPersistentConfig& config,
    const RuntimeTupleDescriptor& tuple,
    CpuInitialization& initialized) noexcept
{
  return InitializeInternal(config, tuple, nullptr, initialized);
}

bool detail::CpuNumericsCore::RunBorrowedEuler(
    const generic::EulerHdgRequest& request) noexcept
{
  if (!InitializeBorrowedEuler(request)) { return false; }
  StateCoordinates coordinates{0.0, 0, 0, 0, 0.0};
  return AdvanceBorrowedEuler(cpu_backward_euler_dt, coordinates);
}

bool detail::CpuNumericsCore::InitializeBorrowedEuler(
    const generic::EulerHdgRequest& request) noexcept
{
  if (!generic::ExactNumericalPolicyConsensus(request.communicator,
                                              request.numerical_policy) ||
      !generic::ValidEulerHdgNumericalPolicy(request.numerical_policy))
  {
    return false;
  }
  CpuPersistentConfig config = MakeCpuSmokeConfig();
  config.polynomial_order = request.polynomial_order;
  // This is not a Cartesian-grid request.  Keep standalone grid provenance
  // inapplicable instead of inventing an elements-per-direction value.
  config.elements_per_direction = 0;
  config.assembly_quadrature_order =
      request.numerical_policy.stabilization.kind ==
              generic::RegularizationKind::igr
          ? request.numerical_policy.igr_assembly_quadrature_order
          : request.numerical_policy.conventional_assembly_quadrature_order;
  // The value-only physical boundary is a frozen-exterior Picard row, so its
  // outer update is intentionally inexact rather than a differentiated
  // Newton step.  Retain the same engine while allowing that convergent outer
  // iteration enough bounded sweeps.
  config.maximum_newton_updates = request.numerical_policy.maximum_nonlinear_updates;
  config.maximum_linear_iterations =
      request.numerical_policy.maximum_linear_iterations;
  config.maximum_line_search_trials =
      request.numerical_policy.maximum_line_search_trials;
  config.armijo_constant = request.numerical_policy.armijo_constant;
  config.nonlinear_tolerance =
      request.numerical_policy.nonlinear_relative_tolerance;
  config.linear_relative_tolerance =
      request.numerical_policy.linear_relative_tolerance;
  config.true_residual_tolerance =
      request.numerical_policy.true_residual_relative_tolerance;
  config.conservation_relative_tolerance =
      request.numerical_policy.conservation_relative_tolerance;
  RuntimeTupleDescriptor tuple{};
  tuple.model = StableIdentity::Copy("euler.conservative.2d.v1");
  tuple.formulation = StableIdentity::Copy("hdg.conventional.v1");
  tuple.trace = StableIdentity::Copy("trace.dg.interface.v1");
  tuple.stabilization = StableIdentity::Copy(
      request.numerical_policy.stabilization.kind == generic::RegularizationKind::igr
          ? "igr.scalar.traced.alpha-beta.v1"
          : ::dg::hdg::runtime::borrowed_euler_llf_stabilization_id);
  tuple.time_treatment = StableIdentity::Copy("backward-euler.v1");
  tuple.backend = StableIdentity::Copy("cpu.host.v1");
  tuple.precision = StableIdentity::Copy("fp64.v1");

  CpuInitialization initialized;
  if (!InitializeInternal(config, tuple, &request, initialized))
  {
    return false;
  }
  SealOwnedAllocationBaseline();
  return true;
}

bool detail::CpuNumericsCore::AuditBorrowedEulerLlfPolicy(
    double& base_stabilization, double& refreshed_stabilization,
    double& frozen_jacobian_relative_error,
    double& refreshed_candidate_relative_difference) noexcept
{
  base_stabilization = std::numeric_limits<double>::infinity();
  refreshed_stabilization = std::numeric_limits<double>::infinity();
  frozen_jacobian_relative_error =
      std::numeric_limits<double>::infinity();
  refreshed_candidate_relative_difference =
      std::numeric_limits<double>::infinity();
  if (!implementation_) { return false; }
  const Implementation& state = *implementation_;
  bool local_valid = state.initialized && state.IsBorrowed() &&
      !state.IsIgr() &&
      state.tuple.stabilization.view() ==
          ::dg::hdg::runtime::borrowed_euler_llf_stabilization_id;
  if (!state.CollectiveAll(local_valid)) { return false; }

  // The trace state owns the maximum at this witness, and both radii are
  // deliberately below the legacy manufactured-fixture value 2.5.
  const State interior{{1.0, 0.10, -0.02, 2.55}};
  const State trace{{0.90, 0.45, 0.03, 2.80}};
  const State interior_direction{{0.04, -0.03, 0.02, 0.05}};
  const State trace_direction{{-0.02, 0.35, -0.01, 0.08}};
  constexpr std::array<double, 2> normal{{1.0, 0.0}};
  double interior_radius = 0.0;
  double trace_radius = 0.0;
  local_valid = NormalSpectralRadius(interior, normal, interior_radius) &&
      NormalSpectralRadius(trace, normal, trace_radius) &&
      BorrowedEulerLlfStabilization(
          interior, trace, normal, base_stabilization) &&
      trace_radius > interior_radius &&
      base_stabilization == std::max(interior_radius, trace_radius) &&
      base_stabilization < cpu_fixed_tau;

  State refreshed_interior{};
  State refreshed_trace{};
  constexpr double candidate_step = 0.25;
  for (int component = 0; component < cpu_conserved_components; ++component)
  {
    refreshed_interior[component] = interior[component] +
        candidate_step * interior_direction[component];
    refreshed_trace[component] = trace[component] +
        candidate_step * trace_direction[component];
  }
  local_valid = local_valid && BorrowedEulerLlfStabilization(
      refreshed_interior, refreshed_trace, normal, refreshed_stabilization) &&
      refreshed_stabilization != base_stabilization;
  if (!state.CollectiveAll(local_valid)) { return false; }

  Matrix4 trace_jacobian{};
  local_valid = NormalFluxJacobian(trace, normal, trace_jacobian);
  State frozen_action{};
  for (int row = 0; row < cpu_conserved_components; ++row)
  {
    frozen_action[row] = base_stabilization *
        (interior_direction[row] - trace_direction[row]);
    for (int column = 0; column < cpu_conserved_components; ++column)
      frozen_action[row] +=
          trace_jacobian[row][column] * trace_direction[column];
  }

  const auto candidate_flux = [](const State& candidate_interior,
                                 const State& candidate_trace,
                                 const std::array<double, 2>& face_normal,
                                 const double frozen_stabilization,
                                 const bool refresh_stabilization,
                                 State& numerical_flux) noexcept
  {
    double stabilization = frozen_stabilization;
    if (refresh_stabilization &&
        !BorrowedEulerLlfStabilization(
            candidate_interior, candidate_trace, face_normal,
            stabilization))
    {
      return false;
    }
    if (!NormalFlux(candidate_trace, face_normal, numerical_flux))
      return false;
    for (int component = 0; component < cpu_conserved_components; ++component)
      numerical_flux[component] += stabilization *
          (candidate_interior[component] - candidate_trace[component]);
    return Finite(numerical_flux);
  };
  const double difference_step =
      std::cbrt(std::numeric_limits<double>::epsilon());
  State plus_interior{}, minus_interior{}, plus_trace{}, minus_trace{};
  for (int component = 0; component < cpu_conserved_components; ++component)
  {
    plus_interior[component] = interior[component] +
        difference_step * interior_direction[component];
    minus_interior[component] = interior[component] -
        difference_step * interior_direction[component];
    plus_trace[component] = trace[component] +
        difference_step * trace_direction[component];
    minus_trace[component] = trace[component] -
        difference_step * trace_direction[component];
  }
  State frozen_plus{}, frozen_minus{}, refreshed_plus{}, refreshed_minus{};
  local_valid = local_valid && candidate_flux(
      plus_interior, plus_trace, normal, base_stabilization, false,
      frozen_plus) && candidate_flux(
      minus_interior, minus_trace, normal, base_stabilization, false,
      frozen_minus) && candidate_flux(
      plus_interior, plus_trace, normal, base_stabilization, true,
      refreshed_plus) && candidate_flux(
      minus_interior, minus_trace, normal, base_stabilization, true,
      refreshed_minus);
  if (!state.CollectiveAll(local_valid)) { return false; }

  double frozen_error_squared = 0.0;
  double refreshed_difference_squared = 0.0;
  double action_squared = 0.0;
  const double inverse_width = 0.5 / difference_step;
  for (int component = 0; component < cpu_conserved_components; ++component)
  {
    const double frozen_difference = inverse_width *
        (frozen_plus[component] - frozen_minus[component]);
    const double refreshed_difference = inverse_width *
        (refreshed_plus[component] - refreshed_minus[component]);
    const double frozen_error =
        frozen_difference - frozen_action[component];
    const double refreshed_error =
        refreshed_difference - frozen_action[component];
    frozen_error_squared += frozen_error * frozen_error;
    refreshed_difference_squared += refreshed_error * refreshed_error;
    action_squared += frozen_action[component] * frozen_action[component];
  }
  const double denominator = std::max(std::sqrt(action_squared),
                                      difference_step);
  frozen_jacobian_relative_error =
      std::sqrt(frozen_error_squared) / denominator;
  refreshed_candidate_relative_difference =
      std::sqrt(refreshed_difference_squared) / denominator;
  local_valid = Finite(frozen_jacobian_relative_error) &&
      Finite(refreshed_candidate_relative_difference) &&
      frozen_jacobian_relative_error <= 1.0e-7 &&
      refreshed_candidate_relative_difference >= 1.0e-4;
  if (!state.CollectiveAll(local_valid)) { return false; }

  std::array<double, 6> assembled_relative_errors{};
  constexpr double assembled_relative_tolerance = 5.0e-5;
  return AuditBorrowedEulerAssemblerJacobian(
      assembled_relative_tolerance, assembled_relative_errors);
}

bool detail::CpuNumericsCore::ArmExactZeroInitialResidualForTests() noexcept
{
  if (!implementation_) { return false; }
  Implementation& state = *implementation_;
  const bool local_valid = state.initialized && state.IsBorrowed() &&
      !state.borrowed_prepare_active &&
      state.active_candidate_generation == 0 &&
      !state.exact_zero_initial_residual_for_tests;
  if (!state.CollectiveAll(local_valid)) { return false; }
  state.exact_zero_initial_residual_for_tests = true;
  return true;
}

bool detail::CpuNumericsCore::AuditBorrowedEulerAssemblerJacobian(
    const double relative_tolerance,
    std::array<double, 6>& relative_errors) noexcept
{
  relative_errors.fill(std::numeric_limits<double>::infinity());
  if (!implementation_) { return false; }
  Implementation& state = *implementation_;
  const bool regularized = state.IsIgr();
  const CpuSlotStorage* const committed = state.Committed();
  bool local_valid = state.initialized && state.IsBorrowed() && committed &&
      state.mesh && state.interior_space && state.trace_space &&
      state.exchange_plan &&
      state.interior_components ==
          (regularized ? igr_interior_components : cpu_conserved_components) &&
      state.trace_components ==
          (regularized ? igr_trace_components : cpu_conserved_components) &&
      committed->interior.size() ==
          static_cast<std::size_t>(state.interior_space->GetVSize()) &&
      committed->trace.size() ==
          static_cast<std::size_t>(state.trace_space->GetTrueVSize()) &&
      Finite(relative_tolerance) && relative_tolerance > 0.0;
  if (!state.CollectiveAll(local_valid)) { return false; }

  std::unique_ptr<mfem::ParGridFunction> base_interior;
  std::unique_ptr<mfem::ParGridFunction> plus_interior;
  std::unique_ptr<mfem::ParGridFunction> minus_interior;
  std::unique_ptr<mfem::ParGridFunction> base_trace;
  std::unique_ptr<mfem::ParGridFunction> plus_trace;
  std::unique_ptr<mfem::ParGridFunction> minus_trace;
  std::unique_ptr<mfem::ParGridFunction> trace_direction;
  mfem::Vector base_true_trace;
  mfem::Vector plus_true_trace;
  mfem::Vector minus_true_trace;
  mfem::Vector interior_direction;
  mfem::Vector true_trace_direction;
  try
  {
    base_interior = std::make_unique<mfem::ParGridFunction>(
        state.interior_space.get());
    plus_interior = std::make_unique<mfem::ParGridFunction>(
        state.interior_space.get());
    minus_interior = std::make_unique<mfem::ParGridFunction>(
        state.interior_space.get());
    base_trace = std::make_unique<mfem::ParGridFunction>(
        state.trace_space.get());
    plus_trace = std::make_unique<mfem::ParGridFunction>(
        state.trace_space.get());
    minus_trace = std::make_unique<mfem::ParGridFunction>(
        state.trace_space.get());
    trace_direction = std::make_unique<mfem::ParGridFunction>(
        state.trace_space.get());
    const int true_trace_size = state.trace_space->GetTrueVSize();
    base_true_trace.SetSize(true_trace_size);
    plus_true_trace.SetSize(true_trace_size);
    minus_true_trace.SetSize(true_trace_size);
    interior_direction.SetSize(state.interior_space->GetVSize());
    true_trace_direction.SetSize(true_trace_size);
    local_valid = base_interior && plus_interior && minus_interior &&
        base_trace && plus_trace && minus_trace && trace_direction &&
        Assign(*base_interior, committed->interior) &&
        Assign(base_true_trace, committed->trace);
  }
  catch (...)
  {
    local_valid = false;
  }
  if (!state.CollectiveAll(local_valid)) { return false; }

  const auto deterministic_unit = [](const std::size_t index,
                                     const std::uint64_t salt) noexcept
  {
    std::uint64_t value =
        (static_cast<std::uint64_t>(index) + 1U) *
            11400714819323198485ULL +
        salt * 7046029254386353131ULL;
    value ^= value >> 29;
    const double magnitude = 0.25 + 0.75 *
        static_cast<double>((value >> 8) % 29U) / 28.0;
    return (value & 1U) == 0U ? magnitude : -magnitude;
  };

  try
  {
    double* const interior_values = base_interior->HostReadWrite();
    double* const trace_values = base_true_trace.HostReadWrite();
    double* const interior_delta = interior_direction.HostWrite();
    double* const trace_delta = true_trace_direction.HostWrite();
    const int interior_size = base_interior->Size();
    const int trace_size = base_true_trace.Size();
    local_valid = (interior_values || interior_size == 0) &&
        (trace_values || trace_size == 0) &&
        (interior_delta || interior_size == 0) &&
        (trace_delta || trace_size == 0) &&
        interior_size % state.interior_components == 0 &&
        trace_size % state.trace_components == 0;
    const int interior_scalar = local_valid
        ? interior_size / state.interior_components : 0;
    const int trace_scalar = local_valid
        ? trace_size / state.trace_components : 0;
    if (regularized)
    {
      for (int scalar = 0; scalar < interior_scalar; ++scalar)
      {
        interior_values[4 * interior_scalar + scalar] =
            0.035 * deterministic_unit(static_cast<std::size_t>(scalar), 11U);
        interior_values[5 * interior_scalar + scalar] =
            0.025 * deterministic_unit(static_cast<std::size_t>(scalar), 17U);
        interior_values[6 * interior_scalar + scalar] =
            0.025 * deterministic_unit(static_cast<std::size_t>(scalar), 23U);
      }
      for (int scalar = 0; scalar < trace_scalar; ++scalar)
        trace_values[4 * trace_scalar + scalar] =
            0.02 * deterministic_unit(static_cast<std::size_t>(scalar), 31U);
    }
    else
    {
      // Keep the Euler trace admissible while making U-Uhat nonzero, so the
      // refreshed-tau finite difference is observably distinct from the
      // frozen-tau Jacobian at this exact linearization.
      for (int scalar = 0; scalar < trace_scalar; ++scalar)
      {
        trace_values[trace_scalar + scalar] +=
            0.08 * deterministic_unit(static_cast<std::size_t>(scalar), 31U);
        trace_values[3 * trace_scalar + scalar] +=
            0.04 * deterministic_unit(static_cast<std::size_t>(scalar), 37U);
      }
    }
    for (int index = 0; index < interior_size; ++index)
      interior_delta[index] =
          deterministic_unit(static_cast<std::size_t>(index), 41U) *
          std::max(1.0, std::abs(interior_values[index]));
    for (int index = 0; index < trace_size; ++index)
      trace_delta[index] =
          deterministic_unit(static_cast<std::size_t>(index), 47U) *
          std::max(1.0, std::abs(trace_values[index]));
    local_valid = local_valid && Finite(*base_interior) &&
        Finite(base_true_trace) && Finite(interior_direction) &&
        Finite(true_trace_direction);
  }
  catch (...)
  {
    local_valid = false;
  }
  if (!state.CollectiveAll(local_valid)) { return false; }

  const bool base_trace_loaded =
      state.LoadTrace(base_true_trace, *base_trace);
  const bool direction_trace_loaded =
      state.LoadTrace(true_trace_direction, *trace_direction);
  if (!base_trace_loaded || !direction_trace_loaded)
  {
    return false;
  }

  const auto& numerical_policy = state.borrowed_request->numerical_policy;
  const int quadrature_order = regularized
      ? numerical_policy.igr_assembly_quadrature_order
      : numerical_policy.conventional_assembly_quadrature_order;
  std::vector<double> frozen_borrowed_llf;
  local_valid = state.Assemble(*base_interior, *base_trace,
                               state.current_system, quadrature_order, false,
                               nullptr,
                               regularized ? nullptr : &frozen_borrowed_llf);
  local_valid = local_valid &&
      (regularized || (!frozen_borrowed_llf.empty() &&
                       Finite(frozen_borrowed_llf) &&
                       std::all_of(
                           frozen_borrowed_llf.begin(),
                           frozen_borrowed_llf.end(),
                           [](const double stabilization)
                           { return stabilization < cpu_fixed_tau; })));
  if (!state.CollectiveAll(local_valid)) { return false; }

  std::vector<double> matrix_a;
  std::vector<double> matrix_b;
  std::vector<double> matrix_c;
  std::vector<double> matrix_d;
  std::vector<double> a_direction;
  std::vector<double> b_direction;
  std::vector<double> c_direction;
  std::vector<double> d_direction;
  try
  {
    matrix_a = state.current_system.a;
    matrix_b = state.current_system.b;
    matrix_c = state.current_system.c;
    matrix_d = state.current_system.d;
    a_direction.assign(state.current_system.residual_interior.size(), 0.0);
    b_direction.assign(state.current_system.residual_interior.size(), 0.0);
    c_direction.assign(state.current_system.residual_trace.size(), 0.0);
    d_direction.assign(state.current_system.residual_trace.size(), 0.0);
    const double* const interior_delta = interior_direction.HostRead();
    const double* const local_trace_delta = trace_direction->HostRead();
    const std::size_t elements =
        static_cast<std::size_t>(state.mesh->GetNE());
    const std::size_t expected_interior =
        elements * static_cast<std::size_t>(state.local_interior);
    const std::size_t expected_trace =
        elements * static_cast<std::size_t>(state.local_trace);
    local_valid =
        (interior_delta || interior_direction.Size() == 0) &&
        (local_trace_delta || trace_direction->Size() == 0) &&
        a_direction.size() == expected_interior &&
        b_direction.size() == expected_interior &&
        c_direction.size() == expected_trace &&
        d_direction.size() == expected_trace &&
        matrix_a.size() == expected_interior * state.local_interior &&
        matrix_b.size() == expected_interior * state.local_trace &&
        matrix_c.size() == expected_trace * state.local_interior &&
        matrix_d.size() == expected_trace * state.local_trace;
    std::vector<double> element_interior_direction(
        static_cast<std::size_t>(state.local_interior), 0.0);
    std::vector<double> element_trace_direction(
        static_cast<std::size_t>(state.local_trace), 0.0);
    for (int element = 0;
         element < state.mesh->GetNE() && local_valid; ++element)
    {
      for (int local = 0; local < state.local_interior; ++local)
      {
        double sign = 1.0;
        const int dof = Decode(state.element_interior_vdofs[
            static_cast<std::size_t>(element) * state.local_interior + local],
                               sign);
        if (dof < 0 || dof >= interior_direction.Size())
        {
          local_valid = false;
          break;
        }
        element_interior_direction[static_cast<std::size_t>(local)] =
            sign * interior_delta[dof];
      }
      for (int local = 0;
           local < state.local_trace && local_valid; ++local)
      {
        double sign = 1.0;
        const int dof = Decode(state.signed_trace_dofs[
            static_cast<std::size_t>(element) * state.local_trace + local],
                               sign);
        if (dof < 0 || dof >= trace_direction->Size())
        {
          local_valid = false;
          break;
        }
        element_trace_direction[static_cast<std::size_t>(local)] =
            sign * local_trace_delta[dof];
      }
      const std::size_t interior_offset =
          static_cast<std::size_t>(element) * state.local_interior;
      const std::size_t trace_offset =
          static_cast<std::size_t>(element) * state.local_trace;
      const std::size_t a_offset = interior_offset * state.local_interior;
      const std::size_t b_offset =
          static_cast<std::size_t>(element) * state.local_interior *
          state.local_trace;
      const std::size_t c_offset =
          static_cast<std::size_t>(element) * state.local_trace *
          state.local_interior;
      const std::size_t d_offset = trace_offset * state.local_trace;
      for (int row = 0; row < state.local_interior; ++row)
      {
        for (int column = 0; column < state.local_interior; ++column)
          a_direction[interior_offset + row] +=
              matrix_a[a_offset +
                       static_cast<std::size_t>(row) * state.local_interior +
                       column] *
              element_interior_direction[static_cast<std::size_t>(column)];
        for (int column = 0; column < state.local_trace; ++column)
          b_direction[interior_offset + row] +=
              matrix_b[b_offset +
                       static_cast<std::size_t>(row) * state.local_trace +
                       column] *
              element_trace_direction[static_cast<std::size_t>(column)];
      }
      for (int row = 0; row < state.local_trace; ++row)
      {
        for (int column = 0; column < state.local_interior; ++column)
          c_direction[trace_offset + row] +=
              matrix_c[c_offset +
                       static_cast<std::size_t>(row) * state.local_interior +
                       column] *
              element_interior_direction[static_cast<std::size_t>(column)];
        for (int column = 0; column < state.local_trace; ++column)
          d_direction[trace_offset + row] +=
              matrix_d[d_offset +
                       static_cast<std::size_t>(row) * state.local_trace +
                       column] *
              element_trace_direction[static_cast<std::size_t>(column)];
      }
    }
    local_valid = local_valid && Finite(a_direction) && Finite(b_direction) &&
        Finite(c_direction) && Finite(d_direction);
  }
  catch (...)
  {
    local_valid = false;
  }
  if (!state.CollectiveAll(local_valid)) { return false; }

  const double difference_step =
      std::cbrt(std::numeric_limits<double>::epsilon());
  bool refreshed_tau_changed = false;
  const auto copy_with_perturbation = [](
      const mfem::Vector& base, const mfem::Vector& direction,
      const double factor, mfem::Vector& destination) noexcept
  {
    if (base.Size() != direction.Size() || base.Size() != destination.Size())
      return false;
    const double* const base_values = base.HostRead();
    const double* const delta_values = direction.HostRead();
    double* const output = destination.HostWrite();
    if ((!base_values || !delta_values || !output) && base.Size() != 0)
      return false;
    for (int index = 0; index < base.Size(); ++index)
      output[index] = base_values[index] + factor * delta_values[index];
    return Finite(destination);
  };

  const auto central_difference = [&state, &copy_with_perturbation,
      &base_interior, &plus_interior, &minus_interior, &base_true_trace,
      &plus_true_trace, &minus_true_trace, &interior_direction,
      &true_trace_direction, &plus_trace, &minus_trace, quadrature_order,
      difference_step, regularized,
      &frozen_borrowed_llf,
      &refreshed_tau_changed](const bool perturb_interior,
                       const bool perturb_trace, const bool freeze_llf,
                       std::vector<double>& interior_difference,
                       std::vector<double>& trace_difference) noexcept
  {
    bool local_ready = false;
    try
    {
      local_ready = copy_with_perturbation(
          *base_interior, interior_direction,
          perturb_interior ? difference_step : 0.0, *plus_interior) &&
          copy_with_perturbation(
              *base_interior, interior_direction,
              perturb_interior ? -difference_step : 0.0, *minus_interior) &&
          copy_with_perturbation(
              base_true_trace, true_trace_direction,
              perturb_trace ? difference_step : 0.0, plus_true_trace) &&
          copy_with_perturbation(
              base_true_trace, true_trace_direction,
              perturb_trace ? -difference_step : 0.0, minus_true_trace);
    }
    catch (...)
    {
      local_ready = false;
    }
    if (!state.CollectiveAll(local_ready)) { return false; }
    const bool plus_trace_loaded =
        state.LoadTrace(plus_true_trace, *plus_trace);
    const bool minus_trace_loaded =
        state.LoadTrace(minus_true_trace, *minus_trace);
    if (!plus_trace_loaded || !minus_trace_loaded)
    {
      return false;
    }
    std::vector<double> plus_evaluated_llf;
    std::vector<double> minus_evaluated_llf;
    std::vector<double>* const plus_observation =
        !regularized && !freeze_llf ? &plus_evaluated_llf : nullptr;
    std::vector<double>* const minus_observation =
        !regularized && !freeze_llf ? &minus_evaluated_llf : nullptr;
    local_ready = state.Assemble(
        *plus_interior, *plus_trace, state.trial_system, quadrature_order,
        false, regularized || !freeze_llf ? nullptr : &frozen_borrowed_llf,
        plus_observation);
    if (!state.CollectiveAll(local_ready)) { return false; }
    if (plus_observation)
    {
      refreshed_tau_changed =
          plus_evaluated_llf.size() == frozen_borrowed_llf.size() &&
          !std::equal(plus_evaluated_llf.begin(), plus_evaluated_llf.end(),
                      frozen_borrowed_llf.begin());
    }
    std::vector<double> plus_interior_residual;
    std::vector<double> plus_trace_residual;
    try
    {
      plus_interior_residual = state.trial_system.residual_interior;
      plus_trace_residual = state.trial_system.residual_trace;
    }
    catch (...)
    {
      local_ready = false;
    }
    if (!state.CollectiveAll(local_ready)) { return false; }
    local_ready = state.Assemble(
        *minus_interior, *minus_trace, state.trial_system, quadrature_order,
        false, regularized || !freeze_llf ? nullptr : &frozen_borrowed_llf,
        minus_observation);
    if (!state.CollectiveAll(local_ready)) { return false; }
    if (minus_observation)
    {
      refreshed_tau_changed = refreshed_tau_changed ||
          (minus_evaluated_llf.size() == frozen_borrowed_llf.size() &&
           !std::equal(minus_evaluated_llf.begin(),
                       minus_evaluated_llf.end(),
                       frozen_borrowed_llf.begin()));
    }
    try
    {
      interior_difference.resize(plus_interior_residual.size());
      trace_difference.resize(plus_trace_residual.size());
      const double inverse_width = 0.5 / difference_step;
      for (std::size_t index = 0;
           index < interior_difference.size(); ++index)
        interior_difference[index] = inverse_width *
            (plus_interior_residual[index] -
             state.trial_system.residual_interior[index]);
      for (std::size_t index = 0; index < trace_difference.size(); ++index)
        trace_difference[index] = inverse_width *
            (plus_trace_residual[index] -
             state.trial_system.residual_trace[index]);
      local_ready = Finite(interior_difference) && Finite(trace_difference);
    }
    catch (...)
    {
      local_ready = false;
    }
    return state.CollectiveAll(local_ready);
  };

  std::vector<double> difference_a;
  std::vector<double> difference_c;
  std::vector<double> difference_b;
  std::vector<double> difference_d;
  std::vector<double> difference_combined_interior;
  std::vector<double> difference_combined_trace;
  const bool interior_difference_ready =
      central_difference(true, false, true, difference_a, difference_c);
  const bool trace_difference_ready =
      central_difference(false, true, true, difference_b, difference_d);
  const bool combined_difference_ready = central_difference(
      true, true, true, difference_combined_interior,
      difference_combined_trace);
  if (!interior_difference_ready || !trace_difference_ready ||
      !combined_difference_ready)
  {
    return false;
  }
  std::vector<double> refreshed_combined_interior;
  std::vector<double> refreshed_combined_trace;
  const bool refreshed_difference_ready = regularized || central_difference(
      true, true, false, refreshed_combined_interior,
      refreshed_combined_trace);
  if (!refreshed_difference_ready) { return false; }

  std::vector<double> combined_interior;
  std::vector<double> combined_trace;
  try
  {
    combined_interior.resize(a_direction.size());
    combined_trace.resize(c_direction.size());
    for (std::size_t index = 0; index < combined_interior.size(); ++index)
      combined_interior[index] = a_direction[index] + b_direction[index];
    for (std::size_t index = 0; index < combined_trace.size(); ++index)
      combined_trace[index] = c_direction[index] + d_direction[index];
    local_valid = Finite(combined_interior) && Finite(combined_trace);
  }
  catch (...)
  {
    local_valid = false;
  }
  if (!state.CollectiveAll(local_valid)) { return false; }

  const auto relative_error = [&state, difference_step](
      const std::vector<double>& difference,
      const std::vector<double>& linearized,
      double& error) noexcept
  {
    const bool local_sizes_match = difference.size() == linearized.size();
    if (!state.CollectiveAll(local_sizes_match)) { return false; }
    std::array<double, 4> local{};
    for (std::size_t index = 0; index < difference.size(); ++index)
    {
      const double delta = difference[index] - linearized[index];
      local[0] += delta * delta;
      local[1] += difference[index] * difference[index];
      local[2] += linearized[index] * linearized[index];
    }
    local[3] = static_cast<double>(difference.size());
    std::array<double, 4> global{};
    if (MPI_Allreduce(local.data(), global.data(),
                      static_cast<int>(global.size()), MPI_DOUBLE, MPI_SUM,
                      state.communicator) != MPI_SUCCESS ||
        !Finite(global) || global[0] < 0.0 || global[1] < 0.0 ||
        global[2] < 0.0 || global[3] < 0.0)
    {
      return false;
    }
    const double denominator = std::max({
        std::sqrt(global[1]), std::sqrt(global[2]),
        difference_step * std::sqrt(std::max(1.0, global[3]))});
    error = std::sqrt(global[0]) / denominator;
    return Finite(error);
  };

  const bool a_valid =
      relative_error(difference_a, a_direction, relative_errors[0]);
  const bool b_valid =
      relative_error(difference_b, b_direction, relative_errors[1]);
  const bool c_valid =
      relative_error(difference_c, c_direction, relative_errors[2]);
  const bool d_valid =
      relative_error(difference_d, d_direction, relative_errors[3]);
  const bool combined_interior_valid = relative_error(
      difference_combined_interior, combined_interior, relative_errors[4]);
  const bool combined_trace_valid = relative_error(
      difference_combined_trace, combined_trace, relative_errors[5]);
  double refreshed_interior_difference = 0.0;
  double refreshed_trace_difference = 0.0;
  const bool refreshed_policy_distinguished = regularized ||
      (refreshed_tau_changed &&
       relative_error(refreshed_combined_interior, combined_interior,
                      refreshed_interior_difference) &&
       relative_error(refreshed_combined_trace, combined_trace,
                      refreshed_trace_difference) &&
       std::max(refreshed_interior_difference,
                refreshed_trace_difference) >= 1.0e-4);
  local_valid = a_valid && b_valid && c_valid && d_valid &&
      combined_interior_valid && combined_trace_valid &&
      refreshed_policy_distinguished;
  if (!state.CollectiveAll(local_valid)) { return false; }

  // Leave the reusable primary system at the unperturbed linearization.  No
  // state slot, trace vector, candidate authority, or solver lifecycle state
  // is published by this test-only check.
  local_valid = state.Assemble(*base_interior, *base_trace,
                               state.current_system, quadrature_order, false);
  if (!state.CollectiveAll(local_valid)) { return false; }
  return state.CollectiveAll(std::all_of(
      relative_errors.begin(), relative_errors.end(),
      [relative_tolerance](const double error)
      { return Finite(error) && error <= relative_tolerance; }));
}

bool detail::CpuNumericsCore::AuditLastBorrowedEulerConservation(
    double& maximum_error, double& maximum_relative_error,
    bool& gate) const noexcept
{
  maximum_error = std::numeric_limits<double>::infinity();
  maximum_relative_error = std::numeric_limits<double>::infinity();
  gate = false;
  if (!implementation_) { return false; }
  const Implementation& state = *implementation_;
  const CpuAttemptWorkspace& workspace = state.active_evidence;
  const bool local_valid = state.initialized && state.IsBorrowed() &&
      workspace.attempt.committed &&
      Finite(workspace.maximum_conservation_error) &&
      workspace.maximum_conservation_error >= 0.0 &&
      Finite(workspace.maximum_conservation_relative_error) &&
      workspace.maximum_conservation_relative_error >= 0.0 &&
      workspace.conservation_gate;
  if (!state.CollectiveAll(local_valid)) { return false; }
  maximum_error = workspace.maximum_conservation_error;
  maximum_relative_error = workspace.maximum_conservation_relative_error;
  gate = workspace.conservation_gate;
  return true;
}

bool detail::CpuNumericsCore::AdvanceBorrowedEuler(
    const double dt, StateCoordinates& coordinates) noexcept
{
  std::uint64_t generation = 0;
  if (!PrepareBorrowedEuler(dt, coordinates, generation))
  {
    return false;
  }
  double maximum_wave_speed = 0.0;
  double backend_stable_dt = 0.0;
  return CommitBorrowedEuler(generation, coordinates, maximum_wave_speed,
                             backend_stable_dt);
}

bool detail::CpuNumericsCore::CommittedBorrowedEulerDiagnostics(
    double& maximum_wave_speed, double& backend_stable_dt) noexcept
{
  if (!implementation_) return false;
  const CpuSlotStorage* const committed =
      implementation_->Committed();
  if (!implementation_->CollectiveAll(committed != nullptr))
  {
    return false;
  }
  return implementation_->ScanBorrowedEulerState(
      *committed, maximum_wave_speed, backend_stable_dt);
}

bool detail::CpuNumericsCore::PrepareBorrowedEuler(
    const double dt, StateCoordinates& coordinates, std::uint64_t& generation) noexcept
{
  generation = 0;
  Implementation& state = *implementation_;
  const bool local_coordinates_valid = state.initialized && state.IsBorrowed() &&
      !state.borrowed_prepare_active &&
      state.active_candidate_generation == 0 &&
      Finite(dt) && dt > 0.0 && Finite(coordinates.time) &&
      coordinates.step != std::numeric_limits<std::uint64_t>::max() &&
      coordinates.generation != std::numeric_limits<std::uint64_t>::max() &&
      coordinates.cursor != std::numeric_limits<std::uint64_t>::max() &&
      Finite(coordinates.time + dt);
  if (!state.CollectiveAll(local_coordinates_valid)) { return false; }

  unsigned long long local_dt_bits = 0;
  std::memcpy(&local_dt_bits, &dt, sizeof(local_dt_bits));
  unsigned long long minimum_dt_bits = 0;
  unsigned long long maximum_dt_bits = 0;
  const int minimum_status = MPI_Allreduce(
      &local_dt_bits, &minimum_dt_bits, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
      state.communicator);
  const int maximum_status = MPI_Allreduce(
      &local_dt_bits, &maximum_dt_bits, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
      state.communicator);
  const bool consensus = minimum_status == MPI_SUCCESS &&
      maximum_status == MPI_SUCCESS && minimum_dt_bits == maximum_dt_bits;
  if (!state.CollectiveAll(consensus)) { return false; }

  // The candidate slot must begin as a committed copy.  Execute then writes
  // only that candidate; publication is deferred to CommitBorrowedEuler.
  const bool candidate_refreshed =
      CopySlot(state.committed_slot, state.candidate_slot);
  if (!state.CollectiveAll(candidate_refreshed))
  {
    const bool restored = CopySlot(state.committed_slot, state.candidate_slot);
    InvalidateCandidateGeneration();
    (void)state.CollectiveAll(restored);
    return false;
  }

  const double previous_dt = state.config.dt;
  state.config.dt = dt;
  const AttemptIdentity attempt{coordinates.cursor + 1, coordinates.step,
                                coordinates.generation, 0, dt};
  const CpuNumericalResult numerical = Execute(attempt, coordinates, {});
  const bool accepted = numerical.gates.All() &&
      numerical.originating_failure == FailureReason::none &&
      CpuStateStoragePolicy::CanPublish(state.committed_slot,
                                        state.candidate_slot);
  if (!state.CollectiveAll(accepted))
  {
    state.config.dt = previous_dt;
    const bool reset = CopySlot(state.committed_slot, state.candidate_slot);
    InvalidateCandidateGeneration();
    (void)state.CollectiveAll(reset);
    return false;
  }
  state.borrowed_prepare_active = true;
  state.borrowed_prepared_previous_dt = previous_dt;
  generation = state.active_candidate_generation;
  std::uint64_t minimum_generation = 0;
  std::uint64_t maximum_generation = 0;
  const int generation_minimum_status = MPI_Allreduce(
      &generation, &minimum_generation, 1, MPI_UINT64_T, MPI_MIN,
      state.communicator);
  const int generation_maximum_status = MPI_Allreduce(
      &generation, &maximum_generation, 1, MPI_UINT64_T, MPI_MAX,
      state.communicator);
  if (!state.CollectiveAll(
          generation_minimum_status == MPI_SUCCESS &&
          generation_maximum_status == MPI_SUCCESS && generation != 0 &&
          minimum_generation == maximum_generation))
  {
    state.config.dt = previous_dt;
    const bool reset = CopySlot(state.committed_slot, state.candidate_slot);
    InvalidateCandidateGeneration();
    state.borrowed_prepare_active = false;
    (void)state.CollectiveAll(reset);
    generation = 0;
    return false;
  }
  return true;
}

bool detail::CpuNumericsCore::CommitBorrowedEuler(
    const std::uint64_t generation, StateCoordinates& coordinates,
    double& maximum_wave_speed, double& backend_stable_dt) noexcept
{
  maximum_wave_speed = 0.0;
  backend_stable_dt = std::numeric_limits<double>::quiet_NaN();
  Implementation& state = *implementation_;
  std::uint64_t minimum_generation = 0;
  std::uint64_t maximum_generation = 0;
  const int minimum_status = MPI_Allreduce(
      &generation, &minimum_generation, 1, MPI_UINT64_T, MPI_MIN,
      state.communicator);
  const int maximum_status = MPI_Allreduce(
      &generation, &maximum_generation, 1, MPI_UINT64_T, MPI_MAX,
      state.communicator);
  const bool generation_consensus = minimum_status == MPI_SUCCESS &&
      maximum_status == MPI_SUCCESS && minimum_generation == maximum_generation;
  const bool local_valid = state.initialized && state.IsBorrowed() &&
      state.borrowed_prepare_active && CandidateGenerationActive(generation) &&
      ::dg::hdg::runtime::detail::SameCoordinates(
          coordinates, state.active_evidence.attempt.entry_coordinates) &&
      CpuStateStoragePolicy::CanPublish(state.committed_slot,
                                        state.candidate_slot) &&
      Finite(state.active_evidence.attempt.maximum_wave_speed) &&
      state.active_evidence.attempt.maximum_wave_speed >= 0.0;
  if (!state.CollectiveAll(generation_consensus && local_valid))
  {
    return false;
  }

  const AttemptIdentity attempt = state.active_evidence.attempt.identity;
  const StateHash candidate_hash = state.active_candidate_state_hash;
  if (!state.CollectiveAll(
          CanAuthorizePublicationChain(attempt, generation, candidate_hash)))
  {
    return false;
  }
  // Scan the candidate immediately before it becomes committed.  The slots
  // exchange atomically below, so these trace-inclusive diagnostics describe
  // exactly the bytes that the successful commit publishes; no fallible work
  // is introduced after the final publication authority has been consumed.
  const CpuSlotStorage* const candidate = state.Candidate();
  if (!state.CollectiveAll(candidate != nullptr))
  {
    return false;
  }
  if (!state.ScanBorrowedEulerState(*candidate, maximum_wave_speed,
                                    backend_stable_dt))
  {
    return false;
  }
  // CanAuthorizePublicationChain performed the final collective preflight.
  // No fallible operation occurs between consuming this authority and publish.
  if (!AuthorizePublicationChain(attempt, generation, candidate_hash))
  {
    return false;
  }
  CpuStateStoragePolicy::Publish(state.committed_slot, state.candidate_slot);
  coordinates.time += attempt.dt;
  ++coordinates.step;
  ++coordinates.generation;
  coordinates.cursor = attempt.sequence;
  coordinates.dt = attempt.dt;
  state.borrowed_prepare_active = false;
  InvalidateCandidateGeneration();
  Complete({RunStatus::committed, FailureReason::none, true, false},
           coordinates, {});
  return true;
}

bool detail::CpuNumericsCore::AbortBorrowedEuler(
    const std::uint64_t generation, const StateCoordinates& coordinates) noexcept
{
  Implementation& state = *implementation_;
  std::uint64_t minimum_generation = 0;
  std::uint64_t maximum_generation = 0;
  const int minimum_status = MPI_Allreduce(
      &generation, &minimum_generation, 1, MPI_UINT64_T, MPI_MIN,
      state.communicator);
  const int maximum_status = MPI_Allreduce(
      &generation, &maximum_generation, 1, MPI_UINT64_T, MPI_MAX,
      state.communicator);
  const bool generation_consensus = minimum_status == MPI_SUCCESS &&
      maximum_status == MPI_SUCCESS && minimum_generation == maximum_generation;
  const bool local_valid = state.initialized && state.IsBorrowed() &&
      state.borrowed_prepare_active && CandidateGenerationActive(generation) &&
      ::dg::hdg::runtime::detail::SameCoordinates(
          coordinates, state.active_evidence.attempt.entry_coordinates);
  if (!state.CollectiveAll(generation_consensus && local_valid))
  {
    return false;
  }
  const bool restored = CopySlot(state.committed_slot, state.candidate_slot);
  if (!state.CollectiveAll(restored)) { return false; }
  state.config.dt = state.borrowed_prepared_previous_dt;
  state.borrowed_prepare_active = false;
  InvalidateCandidateGeneration();
  Complete({RunStatus::terminal_failure, FailureReason::backend_failure,
            false, false}, coordinates, {});
  return true;
}

bool detail::CpuNumericsCore::InitializeInternal(
    const CpuPersistentConfig& config,
    const RuntimeTupleDescriptor& tuple,
    const generic::EulerHdgRequest* const borrowed_request,
    CpuInitialization& initialized) noexcept
{
  initialized = {};
  Implementation& state = *implementation_;
  state.borrowed_request = borrowed_request;
  state.communicator = borrowed_request ? borrowed_request->communicator
                                        : MPI_COMM_WORLD;
  if (!state.CollectiveAll(!state.initialized)) { return false; }
  int mpi_initialized = 0;
  int mpi_finalized = 0;
  MPI_Initialized(&mpi_initialized);
  MPI_Finalized(&mpi_finalized);
  const bool generic = borrowed_request != nullptr;
  state.igr_active = generic &&
      borrowed_request->numerical_policy.stabilization.kind ==
          ::dg::hdg::generic::RegularizationKind::igr;
  state.interior_components = state.igr_active
      ? igr_interior_components : cpu_conserved_components;
  state.trace_components = state.igr_active
      ? igr_trace_components : cpu_conserved_components;
  const bool config_consensus = !generic && mpi_initialized != 0 && mpi_finalized == 0
      ? ExactConfigConsensus(config) : false;
  bool local_ok = mpi_initialized != 0 && mpi_finalized == 0 &&
      MPI_Comm_rank(state.communicator, &state.rank) == MPI_SUCCESS &&
      MPI_Comm_size(state.communicator, &state.ranks) == MPI_SUCCESS &&
      (generic ? state.ranks >= 1 && borrowed_request->mesh != nullptr
               : (state.ranks == 1 || state.ranks == 2) && ExactConfig(config) &&
                     ExactRankShape(config, state.ranks) && config_consensus &&
                     ExactTuple(tuple, state.ranks));
  if (!state.CollectiveAll(local_ok)) { return false; }
  state.config = config;
  state.tuple = tuple;

  std::optional<mfem::Mesh> periodic_mesh;
  if (!generic) try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        config.elements_per_direction, config.elements_per_direction,
        mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(cpu_dimension));
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    periodic_mesh.emplace(mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations)));
    for (int element = 0; element < periodic_mesh->GetNE(); ++element)
    {
      // MFEM's two-argument constructor owns partition selection.  The
      // serial attribute is only a stable parent-cell provenance tag.
      periodic_mesh->SetAttribute(element, element + 1);
    }
    local_ok = periodic_mesh.has_value() && periodic_mesh->GetNE() > 0;
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  try
  {
    if (generic)
    {
      state.mesh = const_cast<mfem::ParMesh*>(borrowed_request->mesh);
      // Generic admission already establishes that the communicator has at
      // least one element globally. Individual ranks may legitimately own no
      // elements and must still join the trace/HYPRE collectives below.
      local_ok = state.mesh != nullptr;
    }
    else
    {
      state.owned_mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD,
                                                          *periodic_mesh);
      state.mesh = state.owned_mesh.get();
      for (int element = 0; element < periodic_mesh->GetNE(); ++element)
        periodic_mesh->SetAttribute(element, 1);
      local_ok = state.ObserveWrapper(state.owned_mesh) && state.mesh->GetNE() > 0;
    }
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  local_ok = generic || state.ObserveParentCellIds();
  if (!state.CollectiveAll(local_ok)) { return false; }

  try
  {
    state.interior_collection = std::make_unique<mfem::L2_FECollection>(
        config.polynomial_order, cpu_dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    state.trace_collection =
        std::make_unique<mfem::DG_Interface_FECollection>(
            config.polynomial_order, cpu_dimension,
            mfem::FiniteElement::VALUE,
            mfem::BasisType::GaussLegendre);
    local_ok = state.ObserveWrapper(state.interior_collection) &&
        state.ObserveWrapper(state.trace_collection);
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  try
  {
    state.interior_space = std::make_unique<mfem::ParFiniteElementSpace>(
        state.mesh, state.interior_collection.get(),
        state.interior_components, mfem::Ordering::byNODES);
    local_ok = state.ObserveWrapper(state.interior_space) &&
        (generic || state.interior_space->GetVSize() > 0);
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  try
  {
    state.trace_space = std::make_unique<mfem::ParFiniteElementSpace>(
        state.mesh, state.trace_collection.get(),
        state.trace_components, mfem::Ordering::byNODES);
    local_ok = state.ObserveWrapper(state.trace_space) &&
        (generic || (state.trace_space->GetVSize() > 0 &&
                     state.trace_space->GetTrueVSize() > 0));
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  try
  {
    // This HDG formulation couples ranks through the trace exchange plan; it
    // never reads MFEM face-neighbour interior values.  Avoid materializing
    // that optional MFEM storage for borrowed meshes, where an otherwise
    // valid zero-element rank has no compatible local face-neighbour table.
    if (!generic) { state.interior_space->ExchangeFaceNbrData(); }
    local_ok = true;
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  try
  {
    // See the interior-space rationale above.  The owned standalone route
    // retains its established MFEM face-neighbour setup unchanged.
    if (!generic) { state.trace_space->ExchangeFaceNbrData(); }
    local_ok = true;
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  try
  {
    state.exchange_plan.emplace();
    local_ok = trace::BuildDGInterfaceTraceExchangePlan(
        *state.trace_space, *state.exchange_plan);
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  local_ok = state.exchange_plan && state.exchange_plan->IsConsistent();
  if (!state.CollectiveAll(local_ok)) { return false; }

  local_ok = state.BuildTopology();
  if (!state.CollectiveAll(local_ok)) { return false; }

  try
  {
    const std::size_t interior_size =
        static_cast<std::size_t>(state.interior_space->GetVSize());
    const std::size_t trace_size =
        static_cast<std::size_t>(state.trace_space->GetTrueVSize());
    state.committed_slot = MakeSlot(interior_size, trace_size,
                                    state.allocations);
    state.candidate_slot = MakeSlot(interior_size, trace_size,
                                    state.allocations);
    local_ok = ResizeOwned(state.first_iterate_interior, interior_size,
                           CpuOwnedAllocationSite::recovery_vector,
                           state.allocations) &&
        ResizeOwned(state.first_iterate_trace, trace_size,
                    CpuOwnedAllocationSite::recovery_vector,
                    state.allocations);
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }
  const bool workspace_ready = state.AllocateWorkspace();
  if (!state.CollectiveAll(workspace_ready)) { return false; }

  try
  {
    const bool w8c_config = IsW8cConfig(config);
    const bool smooth_transition = IsSmoothTransitionConfig(config);
    if (generic)
    {
      local_ok = state.FillBorrowedField(
          *state.old_interior, borrowed_request->initial_state) &&
          state.FillBorrowedField(*state.trial_interior,
                                  borrowed_request->initial_state) &&
          state.FillBorrowedField(*state.external_source,
                                  borrowed_request->source);
    }
    else if (smooth_transition)
    {
      const auto project_smooth = [&](mfem::ParGridFunction& field)
      {
        field = 0.0;
        for (int element = 0; element < state.mesh->GetNE(); ++element)
        {
          const mfem::FiniteElement* finite_element =
              state.interior_space->GetFE(element);
          mfem::ElementTransformation* transformation =
              state.mesh->GetElementTransformation(element);
          if (!finite_element || !transformation) { return false; }
          const int scalar_dofs = finite_element->GetDof();
          std::vector<double> mass(
              static_cast<std::size_t>(scalar_dofs) * scalar_dofs, 0.0);
          std::array<std::vector<double>, cpu_conserved_components> rhs;
          for (auto& values : rhs)
            values.assign(static_cast<std::size_t>(scalar_dofs), 0.0);
          mfem::Vector shape(scalar_dofs), point(cpu_dimension);
          const mfem::IntegrationRule& rule = mfem::IntRules.Get(
              finite_element->GetGeomType(),
              w8c::SmoothTransitionV2Case::projection_q);
          for (int q = 0; q < rule.GetNPoints(); ++q)
          {
            const mfem::IntegrationPoint& ip = rule.IntPoint(q);
            transformation->SetIntPoint(&ip);
            finite_element->CalcShape(ip, shape);
            transformation->Transform(ip, point);
            State value{};
            if (!SmoothTransitionConservative(point[0], value)) return false;
            const double weight = ip.weight * transformation->Weight();
            for (int row = 0; row < scalar_dofs; ++row)
            {
              for (int column = 0; column < scalar_dofs; ++column)
              {
                mass[static_cast<std::size_t>(row) * scalar_dofs + column] +=
                    weight * shape[row] * shape[column];
              }
              for (int component = 0;
                   component < cpu_conserved_components; ++component)
              {
                rhs[static_cast<std::size_t>(component)]
                   [static_cast<std::size_t>(row)] +=
                    weight * shape[row] * value[component];
              }
            }
          }
          ProjectionDenseLu factor;
          if (!factor.Factor(mass, scalar_dofs)) return false;
          mfem::Vector local(cpu_conserved_components * scalar_dofs);
          for (int component = 0;
               component < cpu_conserved_components; ++component)
          {
            std::vector<double> coefficients;
            if (!factor.Solve(rhs[static_cast<std::size_t>(component)],
                              coefficients))
            {
              return false;
            }
            for (int scalar = 0; scalar < scalar_dofs; ++scalar)
            {
              local[component * scalar_dofs + scalar] =
                  coefficients[static_cast<std::size_t>(scalar)];
            }
          }
          mfem::Array<int> vdofs;
          mfem::DofTransformation dof_transformation(
              state.interior_space->GetVDim(),
              state.interior_space->GetOrdering());
          state.interior_space->GetElementVDofs(
              element, vdofs, dof_transformation);
          if (vdofs.Size() != local.Size()) { return false; }
          dof_transformation.TransformPrimal(local);
          field.SetSubVector(vdofs, local);
        }
        return Finite(field);
      };
      local_ok = project_smooth(*state.old_interior) &&
          project_smooth(*state.trial_interior);
    }
    else
    {
    auto exact_function = [w8c_config](const mfem::Vector& point,
                                      mfem::Vector& value)
    {
      State conservative{};
      if (!(w8c_config ? W8cConservative(point[0], conservative)
                       : ManufacturedConservative(
                             point[0], point[1], conservative)))
      {
        throw std::runtime_error("exact state projection failed");
      }
      value.SetSize(cpu_conserved_components);
      for (int component = 0; component < cpu_conserved_components;
           ++component)
      {
        value[component] = conservative[component];
      }
    };
    auto perturbed_function = [w8c_config](const mfem::Vector& point,
                                           mfem::Vector& value)
    {
      State conservative{};
      if (!(w8c_config ? W8cConservative(point[0], conservative)
                       : PerturbedConservative(
                             point[0], point[1], conservative)))
      {
        throw std::runtime_error("perturbed state projection failed");
      }
      value.SetSize(cpu_conserved_components);
      for (int component = 0; component < cpu_conserved_components;
           ++component)
      {
        value[component] = conservative[component];
      }
    };
    mfem::VectorFunctionCoefficient exact_coefficient(
        cpu_conserved_components, exact_function);
    mfem::VectorFunctionCoefficient perturbed_coefficient(
        cpu_conserved_components, perturbed_function);
    state.old_interior->ProjectCoefficient(exact_coefficient);
    state.trial_interior->ProjectCoefficient(perturbed_coefficient);
    local_ok = true;
    }
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!state.CollectiveAll(local_ok)) { return false; }

  local_ok = Copy(*state.old_interior, state.Committed()->interior) &&
      Copy(*state.trial_interior, state.first_iterate_interior);
  if (!state.CollectiveAll(local_ok)) { return false; }
  // Trace provenance is one collective mode, never a rank-local branch.  The
  // compatibility empty view selects derived projection only when *every*
  // participating rank supplied an empty view.  Once any rank supplies a
  // trace, every rank must provide its exact local true-vector extent (which
  // can legitimately be zero).  Populate and collectively validate both
  // initialization staging vectors before allowing any rank into either
  // collective trace distribution below.
  bool derive_initial_true_trace = !generic;
  if (generic)
  {
    derive_initial_true_trace = state.CollectiveAll(
        borrowed_request->initial_true_trace.scalar_count == 0);
  }
  const bool iterate_trace_ready = derive_initial_true_trace
      ? state.ProjectTrace(*state.iterate_trace, state.iterate_true_trace,
                           false)
      : state.FillBorrowedTrueTrace(state.iterate_true_trace,
                                    borrowed_request->initial_true_trace);
  const bool trial_trace_ready = derive_initial_true_trace
      ? state.ProjectTrace(*state.trial_trace, state.trial_true_trace, true)
      : state.FillBorrowedTrueTrace(state.trial_true_trace,
                                    borrowed_request->initial_true_trace);
  const bool iterate_trace_admitted = state.CollectiveAll(iterate_trace_ready);
  const bool trial_trace_admitted = state.CollectiveAll(trial_trace_ready);
  if (!iterate_trace_admitted || !trial_trace_admitted)
  {
    return false;
  }
  if (!state.LoadTrace(state.iterate_true_trace, *state.iterate_trace))
  {
    return false;
  }
  local_ok = Copy(state.iterate_true_trace, state.Committed()->trace);
  if (!state.CollectiveAll(local_ok)) { return false; }
  if (!state.LoadTrace(state.trial_true_trace, *state.trial_trace))
  {
    return false;
  }
  local_ok = Copy(state.trial_true_trace, state.first_iterate_trace) &&
      CopySlot(state.committed_slot, state.candidate_slot);
  if (!state.CollectiveAll(local_ok)) { return false; }

  // Prove the exact storage semantics required by the accepted two-phase
  // contract using distinct old/new bytes, then restore the initialization
  // state before the contract observes either slot.
  const StateHash old_interior = HashDoubles(state.Committed()->interior);
  const StateHash old_trace = HashDoubles(state.Committed()->trace);
  const std::uintptr_t committed_backing_before =
      reinterpret_cast<std::uintptr_t>(
          state.committed_slot.interior.cell->backing.get());
  const std::uintptr_t candidate_backing_before =
      reinterpret_cast<std::uintptr_t>(
          state.candidate_slot.interior.cell->backing.get());
  std::copy(state.first_iterate_interior.begin(),
            state.first_iterate_interior.end(),
            state.Candidate()->interior.begin());
  std::copy(state.first_iterate_trace.begin(), state.first_iterate_trace.end(),
            state.Candidate()->trace.begin());
  const StateHash new_interior = HashDoubles(state.Candidate()->interior);
  const StateHash new_trace = HashDoubles(state.Candidate()->trace);
  CpuStateStoragePolicy::Publish(state.committed_slot, state.candidate_slot);
  const bool complete_exchange =
      HashDoubles(state.Committed()->interior) == new_interior &&
      HashDoubles(state.Committed()->trace) == new_trace;
  const bool candidate_retains_old =
      HashDoubles(state.Candidate()->interior) == old_interior &&
      HashDoubles(state.Candidate()->trace) == old_trace;
  const bool disjoint_after_publish = CpuStateStoragePolicy::Disjoint(
      state.committed_slot, state.candidate_slot);
  CpuStateStoragePolicy::RevertPublish(state.committed_slot,
                                       state.candidate_slot);
  const bool exact_inverse =
      HashDoubles(state.Committed()->interior) == old_interior &&
      HashDoubles(state.Committed()->trace) == old_trace &&
      HashDoubles(state.Candidate()->interior) == new_interior &&
      HashDoubles(state.Candidate()->trace) == new_trace &&
      reinterpret_cast<std::uintptr_t>(
          state.committed_slot.interior.cell->backing.get()) ==
          committed_backing_before &&
      reinterpret_cast<std::uintptr_t>(
          state.candidate_slot.interior.cell->backing.get()) ==
          candidate_backing_before;
  const bool disjoint_after_revert = CpuStateStoragePolicy::Disjoint(
      state.committed_slot, state.candidate_slot);
  local_ok = complete_exchange && candidate_retains_old && exact_inverse &&
      disjoint_after_publish && disjoint_after_revert &&
      CopySlot(state.committed_slot, state.candidate_slot);
  if (!state.CollectiveAll(local_ok)) { return false; }

  std::array<int, 3> mesh_identity{{config.polynomial_order,
                                    config.elements_per_direction,
                                    state.mesh->GetNE()}};
  StateHash local_mesh = HashBytes(
      mesh_identity.data(), mesh_identity.size() * sizeof(int));
  if (generic)
  {
    std::vector<int> borrowed_identity;
    try
    {
      borrowed_identity.reserve(static_cast<std::size_t>(state.mesh->GetNE()) * 2);
      for (int element = 0; element < state.mesh->GetNE(); ++element)
      {
        borrowed_identity.push_back(
            static_cast<int>(state.mesh->GetElementBaseGeometry(element)));
        borrowed_identity.push_back(state.mesh->GetAttribute(element));
      }
      local_mesh = HashPair(local_mesh, HashTopologyIntegers(borrowed_identity));
    }
    catch (...) { local_ok = false; }
    if (!state.CollectiveAll(local_ok)) { return false; }
  }
  StateHash local_partition{};
  if (generic)
  {
    std::vector<int> borrowed_partition;
    try
    {
      borrowed_partition.reserve(static_cast<std::size_t>(state.mesh->GetNE()));
      for (int element = 0; element < state.mesh->GetNE(); ++element)
        borrowed_partition.push_back(state.mesh->GetAttribute(element));
      local_partition = HashTopologyIntegers(borrowed_partition);
    }
    catch (...) { local_ok = false; }
    if (!state.CollectiveAll(local_ok)) { return false; }
  }
  else if (state.ranks == 1)
  {
    std::vector<int> serial_partition_compatibility;
    try
    {
      serial_partition_compatibility.assign(
          static_cast<std::size_t>(config.elements_per_direction) *
              config.elements_per_direction,
          0);
    }
    catch (...) { local_ok = false; }
    if (!state.CollectiveAll(local_ok)) { return false; }
    local_partition = HashTopologyIntegers(serial_partition_compatibility);
  }
  else
  {
    local_partition = HashBytes(
        state.local_parent_cell_ids_sorted.data(),
        state.local_parent_cell_ids_sorted.size() * sizeof(std::uint64_t));
  }
  StateHash local_topology = HashPair(
      HashTopologyIntegers(state.signed_trace_dofs),
      HashBytes(state.trace_canonical_ids.data(),
                state.trace_canonical_ids.size() * sizeof(std::uint64_t)));
  const StateHash mesh_hash = CollectiveHash(local_mesh, state.ranks,
                                             state.communicator);
  const StateHash partition_hash =
      CollectiveHash(local_partition, state.ranks, state.communicator);
  const StateHash topology_hash = CollectiveHash(local_topology, state.ranks,
                                                  state.communicator);
  PersistentResourceDescriptor resources;
  resources.mfem_pin = StableIdentity::Copy(expected_mfem_pin);
  resources.mesh_hash = mesh_hash;
  resources.partition_hash = partition_hash;
  resources.topology_hash = topology_hash;
  resources.rank_count = static_cast<std::uint32_t>(state.ranks);
  resources.polynomial_order = config.polynomial_order;
  resources.layout = StableIdentity::Copy(state.IsIgr()
      ? "l2-interior.true-trace-owned.component-major.igr-extension.7x5.v1"
      : detail::cpu_runtime_layout_identity);
  std::size_t persistent_bytes = 0;
  const auto add_persistent = [&persistent_bytes](const std::size_t count) noexcept
  {
    return count <= (std::numeric_limits<std::size_t>::max() - persistent_bytes) /
            sizeof(double) &&
        (persistent_bytes += count * sizeof(double), true);
  };
  bool persistent_bytes_valid =
      add_persistent(state.Committed()->interior.capacity()) &&
      add_persistent(state.Committed()->trace.capacity()) &&
      add_persistent(state.Candidate()->interior.capacity()) &&
      add_persistent(state.Candidate()->trace.capacity());
  if (!state.CollectiveAll(persistent_bytes_valid)) { return false; }
  resources.reuse = {true, true, true, false, 1, 1,
                     persistent_bytes, state.WorkspaceBytes()};
  if (!state.CollectiveAll(resources.Valid())) { return false; }

  state.evidence.tuple = tuple;
  // A borrowed mesh has no standalone scenario or manufactured-source
  // provenance.  Preserve those fields for the frozen path only; the generic
  // path instead carries the observed borrowed mesh/partition/topology hashes.
  state.evidence.experiment = generic ? StableIdentity{} : config.experiment;
  state.evidence.source = generic ? StableIdentity{} : config.source;
  state.evidence.rank = state.rank;
  state.evidence.ranks = state.ranks;
  state.evidence.local_elements = state.mesh->GetNE();
  state.evidence.global_elements = static_cast<int>(SumU64(
      static_cast<std::uint64_t>(state.mesh->GetNE()), state.communicator));
  state.evidence.polynomial_order = config.polynomial_order;
  state.evidence.elements_per_direction = config.elements_per_direction;
  state.evidence.assembly_quadrature_order =
      config.assembly_quadrature_order;
  state.evidence.dt = config.dt;
  state.evidence.local_owned_interior_coefficients =
      state.Committed()->interior.size();
  state.evidence.global_owned_interior_coefficients = SumU64(
      state.evidence.local_owned_interior_coefficients, state.communicator);
  state.evidence.local_owned_true_trace_coefficients =
      state.Committed()->trace.size();
  state.evidence.global_owned_true_trace_coefficients = SumU64(
      state.evidence.local_owned_true_trace_coefficients, state.communicator);
  state.evidence.unique_periodic_faces = generic ? 0 :
      2U * static_cast<std::uint64_t>(config.elements_per_direction) *
      static_cast<std::uint64_t>(config.elements_per_direction);
  state.evidence.negative_signed_trace_dofs = SumU64(
      static_cast<std::uint64_t>(std::count_if(
          state.signed_trace_dofs.begin(), state.signed_trace_dofs.end(),
          [](const int dof) { return dof < 0; })), state.communicator);
  state.evidence.mesh_hash = mesh_hash;
  state.evidence.partition_hash = partition_hash;
  state.evidence.topology_hash = topology_hash;
  state.evidence.initial_interior_hash = old_interior;
  state.evidence.initial_trace_hash = old_trace;
  state.evidence.topology_builds = 1;
  const std::uintptr_t first_token = reinterpret_cast<std::uintptr_t>(
      state.committed_slot.interior.cell->backing.get());
  const std::uintptr_t second_token = reinterpret_cast<std::uintptr_t>(
      state.candidate_slot.interior.cell->backing.get());
  state.evidence.storage.committed_storage_token =
      std::min(first_token, second_token);
  state.evidence.storage.candidate_storage_token =
      std::max(first_token, second_token);
  state.evidence.storage.interior_size =
      state.Committed()->interior.size();
  state.evidence.storage.interior_capacity =
      state.Committed()->interior.capacity();
  state.evidence.storage.trace_size = state.Committed()->trace.size();
  state.evidence.storage.trace_capacity = state.Committed()->trace.capacity();
  state.evidence.storage.workspace_bytes = state.WorkspaceBytes();
  state.evidence.storage.sparse_owned_term_capacity =
      state.owned_schur_terms.capacity();
  state.evidence.storage.sparse_outgoing_term_capacity =
      state.outgoing_schur_terms.capacity();
  state.evidence.storage.sparse_incoming_slot_capacity =
      state.incoming_schur_slots.capacity();
  state.evidence.storage.true_diagonal_nnz = static_cast<std::size_t>(
      state.true_diagonal_view.NumNonZeroElems());
  state.evidence.storage.true_offdiagonal_nnz = static_cast<std::size_t>(
      state.true_offdiagonal_view.NumNonZeroElems());
  state.evidence.storage.sparse_owned_term_token =
      reinterpret_cast<std::uintptr_t>(state.owned_schur_terms.data());
  state.evidence.storage.sparse_outgoing_term_token =
      reinterpret_cast<std::uintptr_t>(state.outgoing_schur_terms.data());
  state.evidence.storage.sparse_incoming_slot_token =
      reinterpret_cast<std::uintptr_t>(state.incoming_schur_slots.data());
  state.evidence.storage.topology_generation = 1;
  state.evidence.storage.storage_generation = 1;
  state.evidence.storage.publish_is_complete_slot_exchange =
      complete_exchange;
  state.evidence.storage.candidate_retains_exact_old_state_after_publish =
      candidate_retains_old;
  state.evidence.storage.revert_publish_is_exact_byte_inverse = exact_inverse;
  state.evidence.storage.publish_and_revert_preserve_disjoint_backing =
      disjoint_after_publish && disjoint_after_revert;

  state.initialized = true;
  initialized.committed = state.committed_slot;
  initialized.candidate = state.candidate_slot;
  initialized.resources = resources;
  initialized.coordinates = {0.0, 0, 0, 0, 0.0};
  return true;
}

detail::CpuNumericalResult detail::CpuNumericsCore::Execute(
    const AttemptIdentity& attempt,
    const StateCoordinates& coordinates,
    const CpuFailureInjection& injection) noexcept
{
  Implementation& state = *implementation_;
  CpuNumericalResult result;
  CpuAttemptWorkspace& workspace = state.active_evidence;
  workspace = {};
  CpuAttemptEvidence& evidence = workspace.attempt;
  state.active_candidate_generation = 0;
  state.active_candidate_state_hash = {};
  state.publication_chain_authority = false;
  evidence.identity = attempt;
  evidence.entry_coordinates = coordinates;
  evidence.candidate_cursor = attempt.sequence;
  evidence.rank = state.rank;
  evidence.ranks = state.ranks;
  evidence.failure_source_rank = -1;
  evidence.transfers = {0, 0, 0, 0, 0, true};
  state.entry_coordinates = coordinates;
  state.entry_interior_hash = HashDoubles(state.Committed()->interior);
  state.entry_trace_hash = HashDoubles(state.Committed()->trace);
  evidence.committed_interior_before = state.entry_interior_hash;
  evidence.committed_trace_before = state.entry_trace_hash;

  auto fail = [&](const CpuFailurePhase phase,
                  const CpuAttemptStatus status =
                      CpuAttemptStatus::runtime_failure,
                  const FailureReason reason = FailureReason::backend_failure)
  {
    evidence.failure_phase = phase;
    evidence.status = status;
    evidence.finite_gate = false;
    evidence.admissibility_gate = false;
    evidence.recovery_gate = false;
    evidence.nonlinear_gate = false;
    evidence.true_residual_gate = false;
    evidence.transfers.allocation_events =
        state.allocations.EventsAfterBaseline();
    result.evidence = evidence;
    result.gates = {false, false, false, false, true, false};
    result.originating_failure = reason;
    return result;
  };

  // The private arm operation established collective agreement; consuming
  // the flag here must add no collective or other work to production runs.
  const bool exact_zero_initial_residual_for_tests =
      state.exact_zero_initial_residual_for_tests;
  state.exact_zero_initial_residual_for_tests = false;

  const bool local_entry_ready = state.initialized && state.Committed() &&
      state.Candidate() &&
      state.Candidate()->interior == state.Committed()->interior &&
      state.Candidate()->trace == state.Committed()->trace &&
      (IsW8cConfig(state.config)
           ? coordinates.step < W8cAcceptedSteps(state.config)
           : (state.config.scenario != CpuScenario::functional ||
              coordinates.step < cpu_functional_accepted_steps));
  if (!state.CollectiveAll(local_entry_ready))
  {
    return fail(CpuFailurePhase::setup, CpuAttemptStatus::invalid_request);
  }
  const bool injection_ready =
      state.InjectionArguments(injection, coordinates.step + 1);
  if (!state.CollectiveAll(injection_ready))
  {
    return fail(CpuFailurePhase::setup, CpuAttemptStatus::invalid_request);
  }

  const bool old_assigned = AssignGridFunction(
      *state.old_interior, state.Committed()->interior);
  bool iterate_assigned = false;
  bool trace_assigned = false;
  if (coordinates.step == 0)
  {
    iterate_assigned = AssignGridFunction(
        *state.iterate_interior, state.first_iterate_interior);
    trace_assigned = Assign(
        state.iterate_true_trace, state.first_iterate_trace);
  }
  else
  {
    iterate_assigned = AssignGridFunction(
        *state.iterate_interior, state.Committed()->interior);
    trace_assigned = Assign(
        state.iterate_true_trace, state.Committed()->trace);
  }
  if (!state.CollectiveAll(old_assigned && iterate_assigned && trace_assigned))
  {
    return fail(CpuFailurePhase::setup);
  }
  const bool trace_loaded =
      state.LoadTrace(state.iterate_true_trace, *state.iterate_trace);
  if (!state.CollectiveAll(trace_loaded)) { return fail(CpuFailurePhase::setup); }

  double initial_density = 0.0;
  double initial_pressure = 0.0;
  double initial_wave = 0.0;
  const bool initial_prescreen = state.Prescreen(
      *state.iterate_interior, *state.iterate_trace, initial_density,
      initial_pressure, initial_wave);
  if (!state.CollectiveAll(initial_prescreen))
  {
    return fail(CpuFailurePhase::assembly);
  }
  if (!state.EnterPhase(injection, CpuFailurePhase::assembly))
  {
    return fail(CpuFailurePhase::assembly,
                CpuAttemptStatus::injected_failure);
  }
  bool assembled = state.Assemble(
      *state.iterate_interior, *state.iterate_trace, state.current_system,
      state.config.assembly_quadrature_order);
  const bool assembly_fault = ConsumeOwnedCollectiveTestFault(
      detail::CpuCollectiveTestFault::assembly_completion, state.rank,
      state.incarnation);
  assembled = assembled && !assembly_fault;
  if (!state.CollectiveAll(assembled))
  {
    return fail(CpuFailurePhase::assembly);
  }
  if (exact_zero_initial_residual_for_tests)
  {
    // Qualification-only control: retain the assembled A/B/C/D blocks while
    // supplying an exact-zero residual and right-hand side to isolate the
    // ordinary attempt lifecycle's zero-update branch.
    std::fill(state.current_system.residual_interior.begin(),
              state.current_system.residual_interior.end(), 0.0);
    std::fill(state.current_system.residual_trace.begin(),
              state.current_system.residual_trace.end(), 0.0);
    std::fill(state.current_system.f.begin(), state.current_system.f.end(),
              0.0);
    std::fill(state.current_system.g.begin(), state.current_system.g.end(),
              0.0);
  }
  ++evidence.work.complete_assemblies;
  state.CompletePhase();
  double residual = state.ResidualNorm(state.current_system);
  if (!Finite(residual) || residual < 0.0)
  {
    return fail(CpuFailurePhase::assembly);
  }
  evidence.initial_nonlinear_residual = residual;
  evidence.final_nonlinear_residual = residual;
  std::uint64_t current_local_generation =
      state.next_artifact_generation++;
  Implementation::ArtifactCapability current_local_capability{
      attempt.sequence, attempt.sequence, current_local_generation,
      Implementation::ArtifactStage::local};

  int updates = 0;
  bool recovered = false;
  bool final_reconstruction_validated = false;
  double maximum_true_residual = 0.0;
  while (true)
  {
    const bool candidate_converged =
        residual <= state.config.nonlinear_tolerance *
            (1.0 + evidence.initial_nonlinear_residual);
    if (!candidate_converged &&
        updates >= state.config.maximum_newton_updates)
    {
      break;
    }
    workspace.current_newton = {};
    CpuNewtonWorkspace& current_newton = workspace.current_newton;
    const bool consumed_local = state.ConsumeArtifact(
        std::move(current_local_capability),
        Implementation::ArtifactStage::local, attempt,
        current_local_generation);
    if (!state.CollectiveAll(consumed_local))
    {
      return fail(CpuFailurePhase::condensation);
    }
    if (!state.EnterPhase(injection, CpuFailurePhase::condensation))
    {
      return fail(CpuFailurePhase::condensation,
                  CpuAttemptStatus::injected_failure);
    }
    StateHash matrix_hash{};
    std::uint64_t matrix_refresh_generation = 0;
    if (!state.CondenseAndAssemble(state.current_system, matrix_hash,
                                   matrix_refresh_generation))
    {
      return fail(CpuFailurePhase::condensation);
    }
    ++evidence.work.complete_condensations;
    state.CompletePhase();
    const std::uint64_t condensed_generation = state.next_artifact_generation++;
    Implementation::ArtifactCapability condensed_capability{
        attempt.sequence, attempt.sequence, condensed_generation,
        Implementation::ArtifactStage::condensed};
    const bool consumed_condensed = state.ConsumeArtifact(
        std::move(condensed_capability),
        Implementation::ArtifactStage::condensed, attempt,
        condensed_generation);
    if (!state.CollectiveAll(consumed_condensed))
    {
      return fail(CpuFailurePhase::trace_solve);
    }

    if (!state.EnterPhase(injection, CpuFailurePhase::trace_solve))
    {
      return fail(CpuFailurePhase::trace_solve,
                  CpuAttemptStatus::injected_failure);
    }
    double true_residual = 0.0;
    int linear_iterations = 0;
    StateHash factor_hash{}, solution_hash{};
    std::uint64_t factor_setup_generation = 0;
    bool factor_setup_current = false;
    const bool trace_solved = state.SolveTrace(
        matrix_refresh_generation, true_residual, linear_iterations,
        factor_hash, solution_hash, factor_setup_generation,
        factor_setup_current);
    if (factor_setup_current)
    {
      ++evidence.work.complete_factor_setups;
    }
    if (!state.CollectiveAll(trace_solved && Finite(state.true_increment)))
    {
      return fail(CpuFailurePhase::trace_solve);
    }
    evidence.linear_iterations += linear_iterations;
    maximum_true_residual = std::max(maximum_true_residual, true_residual);
    ++evidence.work.complete_trace_solves;
    state.CompletePhase();
    const std::uint64_t solve_generation = state.next_artifact_generation++;
    Implementation::ArtifactCapability solve_capability{
        attempt.sequence, attempt.sequence, solve_generation,
        Implementation::ArtifactStage::solve};
    const bool consumed_solve = state.ConsumeArtifact(
        std::move(solve_capability), Implementation::ArtifactStage::solve,
        attempt, solve_generation);
    if (!state.CollectiveAll(consumed_solve))
    {
      return fail(CpuFailurePhase::recovery);
    }
    const std::uint64_t solution_generation = state.next_artifact_generation++;
    Implementation::ArtifactCapability solution_capability{
        attempt.sequence, attempt.sequence, solution_generation,
        Implementation::ArtifactStage::solution};
    const bool consumed_solution = state.ConsumeArtifact(
        std::move(solution_capability),
        Implementation::ArtifactStage::solution, attempt,
        solution_generation);
    if (!state.CollectiveAll(consumed_solution))
    {
      return fail(CpuFailurePhase::recovery);
    }

    if (!state.EnterPhase(injection, CpuFailurePhase::recovery))
    {
      return fail(CpuFailurePhase::recovery,
                  CpuAttemptStatus::injected_failure);
    }
    if (!state.Recover()) { return fail(CpuFailurePhase::recovery); }
    ++evidence.work.complete_recoveries;
    state.CompletePhase();
    recovered = true;
    const std::uint64_t recovery_generation = state.next_artifact_generation++;
    Implementation::ArtifactCapability recovery_capability{
        attempt.sequence, attempt.sequence, recovery_generation,
        Implementation::ArtifactStage::recovery};
    const bool consumed_recovery = state.ConsumeArtifact(
        std::move(recovery_capability),
        Implementation::ArtifactStage::recovery, attempt,
        recovery_generation);
    current_newton.ordered = current_local_generation < condensed_generation &&
        condensed_generation < solve_generation &&
        solve_generation < solution_generation &&
        solution_generation < recovery_generation;
    current_newton.affine_consumption_verified = consumed_local &&
        consumed_condensed && consumed_solve && consumed_solution &&
        consumed_recovery;
    if (!state.CollectiveAll(current_newton.affine_consumption_verified))
    {
      return fail(CpuFailurePhase::recovery);
    }

    if (candidate_converged)
    {
      // A converged candidate is publication-ready only after its own
      // A/B/C/D blocks, condensation, trace solve, and recovery have formed
      // one coherent chain.  Recovery computes (and validates) the final
      // Newton direction but deliberately does not apply another update.
      current_newton.candidate_generation =
          state.next_candidate_generation++;
      current_newton.candidate_state_hash = HashPair(
          HashVector(*state.iterate_interior),
          HashVector(state.iterate_true_trace));
      if (!state.CollectiveAll(
              Finite(*state.iterate_interior) &&
              Finite(state.iterate_true_trace)))
      {
        return fail(CpuFailurePhase::recovery);
      }
      current_newton.accepted = true;
      final_reconstruction_validated = true;
      break;
    }

    bool accepted = false;
    bool admissible_trial = false;
    for (int trial = 0;
         trial < state.config.maximum_line_search_trials; ++trial)
    {
      const double alpha = std::ldexp(1.0, -trial);
      evidence.final_line_search_trials = trial + 1;
      const double armijo_boundary =
          (1.0 - state.config.armijo_constant * alpha) * residual;
      if (!state.MakeTrial(alpha)) { return fail(CpuFailurePhase::recovery); }
      double density = 0.0;
      double pressure = 0.0;
      double wave = 0.0;
      const bool trial_prescreen = state.Prescreen(
          *state.trial_interior, *state.trial_trace, density, pressure, wave);
      if (!state.CollectiveAll(trial_prescreen)) { continue; }
      admissible_trial = true;
      const bool trial_assembled = state.Assemble(
          *state.trial_interior, *state.trial_trace, state.trial_system,
          state.config.assembly_quadrature_order);
      if (!state.CollectiveAll(trial_assembled))
      {
        return fail(CpuFailurePhase::assembly);
      }
      ++evidence.work.complete_assemblies;
      ++evidence.work.trial_reassemblies;
      const double trial_residual = state.ResidualNorm(state.trial_system);
      if (!Finite(trial_residual)) { return fail(CpuFailurePhase::assembly); }
      if (trial_residual <= armijo_boundary)
      {
        if (!state.AcceptTrial()) { return fail(CpuFailurePhase::recovery); }
        residual = trial_residual;
        evidence.minimum_density = density;
        evidence.minimum_pressure = pressure;
        evidence.maximum_wave_speed = wave;
        current_newton.candidate_generation = state.next_candidate_generation++;
        current_newton.candidate_state_hash = HashPair(
            HashVector(*state.iterate_interior),
            HashVector(state.iterate_true_trace));
        if (!state.CollectiveAll(
                Finite(*state.iterate_interior) && Finite(state.iterate_true_trace)))
        {
          return fail(CpuFailurePhase::recovery);
        }
        current_newton.accepted = true;
        accepted = true;
        break;
      }
    }
    if (!state.CollectiveAll(accepted))
    {
      return fail(
          CpuFailurePhase::recovery, CpuAttemptStatus::runtime_failure,
          IsW8cConfig(state.config)
              ? (admissible_trial ? FailureReason::success_gate
                                  : FailureReason::admissibility_gate)
              : FailureReason::backend_failure);
    }
    ++updates;
    ++evidence.work.accepted_newton_updates;
    current_local_generation = state.next_artifact_generation++;
    current_local_capability = Implementation::ArtifactCapability{
        attempt.sequence, attempt.sequence, current_local_generation,
        Implementation::ArtifactStage::local};
  }

  evidence.nonlinear_updates = updates;
  evidence.final_nonlinear_residual = residual;
  evidence.maximum_true_relative_residual = maximum_true_residual;
  const bool nonlinear_converged =
      residual <= state.config.nonlinear_tolerance *
          (1.0 + evidence.initial_nonlinear_residual);
  const std::uint64_t expected_reconstructions =
      static_cast<std::uint64_t>(updates) + 1U;
  const bool factor_setup_census_exact =
      evidence.work.complete_factor_setups ==
          evidence.work.complete_trace_solves &&
      evidence.work.complete_factor_setups ==
          evidence.work.complete_condensations &&
      evidence.work.complete_factor_setups ==
          evidence.work.complete_recoveries &&
      evidence.work.complete_factor_setups == expected_reconstructions &&
      evidence.work.accepted_newton_updates ==
          static_cast<std::uint64_t>(updates);
  if (!state.CollectiveAll(
          nonlinear_converged && recovered &&
          final_reconstruction_validated && factor_setup_census_exact))
  {
    return fail(CpuFailurePhase::recovery);
  }
  const bool candidate_stored = state.StoreCandidate();
  if (!state.CollectiveAll(candidate_stored))
  {
    return fail(CpuFailurePhase::recovery);
  }
  const StateHash stored_candidate_hash = HashPair(
      HashDoubles(state.Candidate()->interior),
      HashDoubles(state.Candidate()->trace));
  const bool retained_candidate_valid = workspace.current_newton.accepted &&
      workspace.current_newton.ordered &&
      workspace.current_newton.affine_consumption_verified &&
      workspace.current_newton.candidate_state_hash == stored_candidate_hash;
  if (!state.CollectiveAll(retained_candidate_valid))
  {
    return fail(CpuFailurePhase::recovery);
  }
  workspace.current_newton.retained_for_publication = true;
  evidence.candidate_generation = workspace.current_newton.candidate_generation;
  state.active_candidate_generation = workspace.current_newton.candidate_generation;
  state.active_candidate_state_hash = stored_candidate_hash;
  state.publication_chain_authority = true;

  if (!state.EnterPhase(injection,
                        CpuFailurePhase::after_recovery_before_commit))
  {
    return fail(CpuFailurePhase::after_recovery_before_commit,
                CpuAttemptStatus::injected_failure);
  }
  state.CompletePhase();
  if (!state.EnterPhase(injection, CpuFailurePhase::diagnostics))
  {
    return fail(CpuFailurePhase::diagnostics,
                CpuAttemptStatus::injected_failure);
  }
  double final_density = 0.0;
  double final_pressure = 0.0;
  double final_wave = 0.0;
  const bool final_prescreen = state.Prescreen(
      *state.iterate_interior, *state.iterate_trace, final_density,
      final_pressure, final_wave);
  if (!state.CollectiveAll(final_prescreen))
  {
    return fail(CpuFailurePhase::diagnostics);
  }
  workspace.maximum_conservation_error = 0.0;
  workspace.maximum_conservation_relative_error = 0.0;
  workspace.conservation_gate = false;
  if (state.IsBorrowed())
  {
    double conservation_error = 0.0;
    double conservation_relative_error = 0.0;
    const bool conservation_balanced = state.ConservationBalance(
        *state.iterate_interior, *state.iterate_trace, attempt.dt,
        conservation_error, conservation_relative_error);
    workspace.maximum_conservation_error = conservation_error;
    workspace.maximum_conservation_relative_error =
        conservation_relative_error;
    workspace.conservation_gate = conservation_balanced &&
        conservation_relative_error <=
            state.config.conservation_relative_tolerance;
    if (!state.CollectiveAll(workspace.conservation_gate))
    {
      return fail(CpuFailurePhase::diagnostics);
    }
  }
  state.CompletePhase();

  evidence.minimum_density = final_density;
  evidence.minimum_pressure = final_pressure;
  evidence.maximum_wave_speed = final_wave;
  evidence.finite_gate = Finite(residual) && Finite(maximum_true_residual) &&
      Finite(state.Candidate()->interior) && Finite(state.Candidate()->trace);
  evidence.admissibility_gate = state.IsBorrowed()
      ? final_density > 0.0 && final_pressure > 0.0 && Finite(final_wave)
      : final_density >= density_gate && final_pressure >= pressure_gate &&
            final_wave < cpu_fixed_tau;
  evidence.recovery_gate = recovered;
  evidence.nonlinear_gate = nonlinear_converged;
  evidence.true_residual_gate = maximum_true_residual <=
      state.config.true_residual_tolerance;
  if (state.config.force_private_scientific_gate_failure)
    evidence.true_residual_gate = false;
  evidence.status = CpuAttemptStatus::success;
  evidence.failure_phase = CpuFailurePhase::none;
  evidence.transfers.allocation_events =
      state.allocations.EventsAfterBaseline();
  if (evidence.transfers.allocation_events != 0)
  {
    return fail(CpuFailurePhase::terminal_evidence);
  }
  result.evidence = evidence;
  result.gates = {evidence.finite_gate, evidence.admissibility_gate,
                  evidence.recovery_gate, evidence.true_residual_gate,
                  true,
                  evidence.nonlinear_gate &&
                      (!state.IsBorrowed() || workspace.conservation_gate)};
  result.originating_failure = FailureReason::none;
  return result;
}

void detail::CpuNumericsCore::Complete(
    const AttemptOutcome& outcome,
    const StateCoordinates& coordinates,
    const RunTelemetry&) noexcept
{
  Implementation& state = *implementation_;
  CpuAttemptEvidence& attempt = state.active_evidence.attempt;
  attempt.exit_coordinates = coordinates;
  attempt.committed_interior_after = HashDoubles(state.Committed()->interior);
  attempt.committed_trace_after = HashDoubles(state.Committed()->trace);
  attempt.committed = outcome.committed;
  attempt.rollback_byte_equal = !outcome.committed &&
      attempt.committed_interior_after == state.entry_interior_hash &&
      attempt.committed_trace_after == state.entry_trace_hash &&
      ::dg::hdg::runtime::detail::SameCoordinates(
          coordinates, state.entry_coordinates);
  if (outcome.committed)
  {
    attempt.status = CpuAttemptStatus::success;
    state.evidence.accepted.valid = true;
    state.evidence.accepted.coordinates = coordinates;
    state.evidence.accepted.interior_hash =
        attempt.committed_interior_after;
    state.evidence.accepted.trace_hash = attempt.committed_trace_after;
    if (!state.IsBorrowed())
    {
      state.evidence.accepted.rank_common_interior_hash =
          CpuStateStoragePolicy::HashInterior(state.committed_slot.interior);
      state.evidence.accepted.rank_common_trace_hash =
          CpuStateStoragePolicy::HashTrace(state.committed_slot.trace);
    }
    state.evidence.accepted.attempt = attempt;
    ++state.evidence.accepted_step_count;
    ++state.evidence.commits;
  }
  else
  {
    if (attempt.status == CpuAttemptStatus::ready ||
        attempt.status == CpuAttemptStatus::success)
    {
      attempt.status = CpuAttemptStatus::runtime_failure;
    }
    if (attempt.failure_phase == CpuFailurePhase::none)
    {
      attempt.failure_phase = CpuFailurePhase::terminal_evidence;
    }
    ++state.evidence.failed_attempts;
  }
  ++state.evidence.attempts;
  if (attempt.identity.retry_index == 1) { ++state.evidence.retries; }
  state.evidence.last_attempt = attempt;
  state.evidence.storage.allocation_events_after_baseline =
      state.allocations.EventsAfterBaseline();
}

bool detail::CpuNumericsCore::CaptureCheckpoint(
    const CheckpointDescriptor& descriptor,
    CpuCheckpoint& checkpoint) noexcept
{
  Implementation& state = *implementation_;
  if (!state.initialized || !descriptor.successful_commit || checkpoint.valid)
  {
    return false;
  }
  CpuStateSlot snapshot;
  CpuOwnedAllocationLedger checkpoint_allocations;
  try
  {
    snapshot = MakeSlot(state.Committed()->interior.size(),
                        state.Committed()->trace.size(),
                        checkpoint_allocations);
  }
  catch (...)
  {
    return false;
  }
  if (!CopySlot(state.committed_slot, snapshot)) { return false; }
  checkpoint.descriptor = descriptor;
  checkpoint.state = std::move(snapshot);
  checkpoint.valid = true;
  return true;
}

void detail::CpuNumericsCore::CompleteRestore(
    const RestoreOutcome& outcome,
    const StateCoordinates& coordinates,
    const RunTelemetry&) noexcept
{
  Implementation& state = *implementation_;
  if (!outcome.restored) { return; }
  CpuAttemptEvidence restored;
  restored.status = CpuAttemptStatus::success;
  restored.exit_coordinates = coordinates;
  restored.committed_interior_after = HashDoubles(state.Committed()->interior);
  restored.committed_trace_after = HashDoubles(state.Committed()->trace);
  restored.committed = true;
  state.evidence.accepted.valid = true;
  state.evidence.accepted.coordinates = coordinates;
  state.evidence.accepted.interior_hash = restored.committed_interior_after;
  state.evidence.accepted.trace_hash = restored.committed_trace_after;
  state.evidence.accepted.rank_common_interior_hash =
      CpuStateStoragePolicy::HashInterior(state.committed_slot.interior);
  state.evidence.accepted.rank_common_trace_hash =
      CpuStateStoragePolicy::HashTrace(state.committed_slot.trace);
  state.evidence.accepted.attempt = restored;
  state.evidence.last_attempt = restored;
}

CpuStateSnapshot detail::CpuNumericsCore::Snapshot(
    const StateCoordinates& coordinates) const noexcept
{
  const Implementation& state = *implementation_;
  CpuStateSnapshot snapshot;
  snapshot.coordinates = coordinates;
  const CpuSlotStorage* storage = state.Committed();
  if (!storage) { return snapshot; }
  snapshot.interior_hash = HashDoubles(storage->interior);
  snapshot.trace_hash = HashDoubles(storage->trace);
  snapshot.state_hash = HashPair(snapshot.interior_hash, snapshot.trace_hash);
  snapshot.interior = {storage->interior.data(), storage->interior.size(),
                       storage->interior.capacity()};
  snapshot.trace = {storage->trace.data(), storage->trace.size(),
                    storage->trace.capacity()};
  snapshot.physical_interior_ids = state.physical_interior_ids.data();
  snapshot.physical_interior_id_count = state.physical_interior_ids.size();
  snapshot.scalar_interior_dofs = state.scalar_interior_dofs;
  return snapshot;
}

CpuStateSnapshot detail::CpuNumericsCore::CandidateSnapshot(
    const StateCoordinates& coordinates) const noexcept
{
  const Implementation& state = *implementation_;
  CpuStateSnapshot snapshot;
  snapshot.coordinates = coordinates;
  const CpuSlotStorage* storage = state.Candidate();
  if (!storage) { return snapshot; }
  snapshot.interior_hash = HashDoubles(storage->interior);
  snapshot.trace_hash = HashDoubles(storage->trace);
  snapshot.state_hash = HashPair(snapshot.interior_hash, snapshot.trace_hash);
  snapshot.interior = {storage->interior.data(), storage->interior.size(),
                       storage->interior.capacity()};
  snapshot.trace = {storage->trace.data(), storage->trace.size(),
                    storage->trace.capacity()};
  snapshot.physical_interior_ids = state.physical_interior_ids.data();
  snapshot.physical_interior_id_count = state.physical_interior_ids.size();
  snapshot.scalar_interior_dofs = state.scalar_interior_dofs;
  return snapshot;
}

void detail::CpuNumericsCore::MarkPreparedFailure(
    const CpuFailurePhase phase, const int source_rank) noexcept
{
  implementation_->active_evidence.attempt.failure_phase = phase;
  implementation_->active_evidence.attempt.failure_source_rank = source_rank;
  implementation_->active_evidence.attempt.status = CpuAttemptStatus::runtime_failure;
}

void detail::CpuNumericsCore::MarkPreparedFailure(
    const CpuFailurePhase phase, const int source_rank,
    const CpuAttemptStatus status) noexcept
{
  implementation_->active_evidence.attempt.failure_phase = phase;
  implementation_->active_evidence.attempt.failure_source_rank = source_rank;
  implementation_->active_evidence.attempt.status = status;
}

bool detail::CpuNumericsCore::AuthorizePublicationChain(
    const AttemptIdentity& attempt,
    const std::uint64_t candidate_generation,
    const StateHash& candidate_state_hash) noexcept
{
  Implementation& state = *implementation_;
  if (!CanAuthorizePublicationChain(
          attempt, candidate_generation, candidate_state_hash))
  {
    return false;
  }
  state.publication_chain_authority = false;
  return true;
}

bool detail::CpuNumericsCore::CanAuthorizePublicationChain(
    const AttemptIdentity& attempt,
    const std::uint64_t candidate_generation,
    const StateHash& candidate_state_hash) const noexcept
{
  const Implementation& state = *implementation_;
  if (!state.publication_chain_authority ||
      state.active_candidate_generation != candidate_generation ||
      state.active_candidate_state_hash != candidate_state_hash ||
      !::dg::hdg::runtime::detail::SameAttempt(
          state.active_evidence.attempt.identity, attempt) ||
      !state.active_evidence.current_newton.retained_for_publication)
  {
    return false;
  }
  const CpuNewtonWorkspace& retained = state.active_evidence.current_newton;
  return retained.ordered && retained.affine_consumption_verified &&
      retained.candidate_generation == candidate_generation &&
      retained.candidate_state_hash == candidate_state_hash;
}

bool detail::CpuNumericsCore::CandidateGenerationActive(
    const std::uint64_t generation) const noexcept
{
  const Implementation& state = *implementation_;
  return generation != 0 && state.active_candidate_generation == generation;
}

void detail::CpuNumericsCore::InvalidateCandidateGeneration() noexcept
{
  Implementation& state = *implementation_;
  if (state.active_candidate_generation != 0)
  {
    state.active_evidence.attempt.candidate_generation_invalidated = true;
  }
  state.active_candidate_generation = 0;
  state.active_candidate_state_hash = {};
  state.publication_chain_authority = false;
}

bool detail::CpuNumericsCore::ObserveOwnedAllocation(
    const CpuOwnedAllocationSite site, const std::size_t bytes,
    const std::size_t alignment, const std::size_t capacity,
    const void* token) noexcept
{
  return implementation_->allocations.Observe(site, bytes, alignment,
                                               capacity, token);
}

void detail::CpuNumericsCore::SealOwnedAllocationBaseline() noexcept
{
  Implementation& state = *implementation_;
  state.allocations.Seal();
  CpuStorageEvidence& storage = state.evidence.storage;
  storage.allocation_baseline = state.allocations.baseline;
  storage.allocation_events_after_baseline =
      state.allocations.EventsAfterBaseline();
  storage.opaque_mfem_hypre_allocations_unobserved = true;
  storage.opaque_standard_library_control_blocks_unobserved = true;
}

const CpuRunEvidence& detail::CpuNumericsCore::evidence() const noexcept
{
  return implementation_->evidence;
}

bool detail::CpuNumericsCore::ExportCommittedInterior(
    double* const destination, const std::size_t size) const noexcept
{
  const Implementation& state = *implementation_;
  const CpuSlotStorage* const committed = state.Committed();
  const std::size_t public_size = committed && state.interior_components > 0
      ? committed->interior.size() /
            static_cast<std::size_t>(state.interior_components) *
            cpu_conserved_components
      : 0;
  if (!state.initialized || !state.IsBorrowed() || !committed ||
      committed->interior.size() % state.interior_components != 0 ||
      size != public_size ||
      (destination == nullptr && size != 0))
  {
    return false;
  }
  if (size != 0)
  {
    std::memcpy(destination, committed->interior.data(),
                size * sizeof(double));
  }
  return true;
}

bool detail::CpuNumericsCore::ExportCommittedInterior(
    mfem::ParGridFunction& destination) const noexcept
{
  const Implementation& state = *implementation_;
  const mfem::FiniteElementSpace* const space = destination.FESpace();
  const CpuSlotStorage* const committed = state.Committed();
  const std::size_t public_size = committed && state.interior_components > 0
      ? committed->interior.size() /
            static_cast<std::size_t>(state.interior_components) *
            cpu_conserved_components
      : 0;
  if (!state.initialized || !state.IsBorrowed() || !space || !committed ||
      space->GetMesh() != state.mesh ||
      space->GetVDim() != cpu_conserved_components ||
      space->GetOrdering() != mfem::Ordering::byNODES ||
      committed->interior.size() % state.interior_components != 0 ||
      destination.Size() != static_cast<int>(public_size))
  {
    return false;
  }
  double* const values = destination.HostWrite();
  if (values == nullptr && destination.Size() != 0) { return false; }
  if (destination.Size() != 0)
  {
    std::memcpy(values, committed->interior.data(),
                public_size * sizeof(double));
  }
  return true;
}

bool detail::CpuNumericsCore::ExportPreparedInterior(
    const std::uint64_t generation, double* const destination,
    const std::size_t size) const noexcept
{
  const Implementation& state = *implementation_;
  std::uint64_t minimum_generation = 0;
  std::uint64_t maximum_generation = 0;
  const int minimum_status = MPI_Allreduce(
      &generation, &minimum_generation, 1, MPI_UINT64_T, MPI_MIN,
      state.communicator);
  const int maximum_status = MPI_Allreduce(
      &generation, &maximum_generation, 1, MPI_UINT64_T, MPI_MAX,
      state.communicator);
  const bool generation_consensus = minimum_status == MPI_SUCCESS &&
      maximum_status == MPI_SUCCESS && minimum_generation == maximum_generation;
  const CpuSlotStorage* const candidate = state.Candidate();
  const std::size_t public_size = candidate && state.interior_components > 0
      ? candidate->interior.size() /
            static_cast<std::size_t>(state.interior_components) *
            cpu_conserved_components
      : 0;
  const bool local_valid = state.initialized && state.IsBorrowed() &&
      state.borrowed_prepare_active && CandidateGenerationActive(generation) &&
      candidate && candidate->interior.size() % state.interior_components == 0 &&
      size == public_size &&
      (destination != nullptr || size == 0);
  if (!state.CollectiveAll(generation_consensus && local_valid))
  {
    return false;
  }
  if (size != 0)
  {
    std::memcpy(destination, candidate->interior.data(),
                size * sizeof(double));
  }
  return true;
}

bool detail::CpuNumericsCore::ExportPreparedInterior(
    const std::uint64_t generation, mfem::ParGridFunction& destination) const noexcept
{
  const Implementation& state = *implementation_;
  const mfem::FiniteElementSpace* const space = destination.FESpace();
  const CpuSlotStorage* const candidate = state.Candidate();
  const std::size_t public_size = candidate && state.interior_components > 0
      ? candidate->interior.size() /
            static_cast<std::size_t>(state.interior_components) *
            cpu_conserved_components
      : 0;
  std::uint64_t minimum_generation = 0;
  std::uint64_t maximum_generation = 0;
  const int minimum_status = MPI_Allreduce(
      &generation, &minimum_generation, 1, MPI_UINT64_T, MPI_MIN,
      state.communicator);
  const int maximum_status = MPI_Allreduce(
      &generation, &maximum_generation, 1, MPI_UINT64_T, MPI_MAX,
      state.communicator);
  const bool generation_consensus = minimum_status == MPI_SUCCESS &&
      maximum_status == MPI_SUCCESS && minimum_generation == maximum_generation;
  const bool local_valid = state.initialized && state.IsBorrowed() &&
      state.borrowed_prepare_active && CandidateGenerationActive(generation) &&
      space && candidate && space->GetMesh() == state.mesh &&
      space->GetVDim() == cpu_conserved_components &&
      space->GetOrdering() == mfem::Ordering::byNODES &&
      candidate->interior.size() % state.interior_components == 0 &&
      destination.Size() == static_cast<int>(public_size);
  if (!state.CollectiveAll(generation_consensus && local_valid))
  {
    return false;
  }
  double* const values = destination.HostWrite();
  if (values == nullptr && destination.Size() != 0) { return false; }
  if (destination.Size() != 0)
  {
    std::memcpy(values, candidate->interior.data(),
                public_size * sizeof(double));
  }
  return true;
}

bool detail::CpuNumericsCore::ExportCanonicalCommitted(
    CpuCanonicalEntry* entries, const std::size_t capacity,
    std::size_t& written) const noexcept
{
  written = 0;
  const Implementation& state = *implementation_;
  const CpuSlotStorage* storage = state.Committed();
  if (!storage || !entries) { return false; }
  const std::size_t required =
      storage->interior.size() + storage->trace.size();
  if (capacity < required ||
      state.interior_global_ids.size() != storage->interior.size() ||
      state.trace_canonical_ids.size() != storage->trace.size())
  {
    return false;
  }
  for (std::size_t index = 0; index < storage->interior.size(); ++index)
  {
    entries[written++] = {state.interior_global_ids[index],
                          storage->interior[index]};
  }
  constexpr std::uint64_t trace_prefix = std::uint64_t{1} << 63;
  for (std::size_t index = 0; index < storage->trace.size(); ++index)
  {
    entries[written++] = {trace_prefix | state.trace_canonical_ids[index],
                          storage->trace[index]};
  }
  std::sort(entries, entries + written,
            [](const CpuCanonicalEntry& left,
               const CpuCanonicalEntry& right)
            { return left.global_id < right.global_id; });
  return true;
}

bool detail::CpuNumericsCore::ExportCanonicalCandidate(
    CpuCanonicalEntry* entries, const std::size_t capacity,
    std::size_t& written) const noexcept
{
  written = 0;
  const Implementation& state = *implementation_;
  const CpuSlotStorage* storage = state.Candidate();
  if (!storage || !entries) { return false; }
  const std::size_t required =
      storage->interior.size() + storage->trace.size();
  if (capacity < required ||
      state.interior_global_ids.size() != storage->interior.size() ||
      state.trace_canonical_ids.size() != storage->trace.size())
  {
    return false;
  }
  for (std::size_t index = 0; index < storage->interior.size(); ++index)
  {
    entries[written++] = {state.interior_global_ids[index],
                          storage->interior[index]};
  }
  constexpr std::uint64_t trace_prefix = std::uint64_t{1} << 63;
  for (std::size_t index = 0; index < storage->trace.size(); ++index)
  {
    entries[written++] = {trace_prefix | state.trace_canonical_ids[index],
                          storage->trace[index]};
  }
  std::sort(entries, entries + written,
            [](const CpuCanonicalEntry& left,
               const CpuCanonicalEntry& right)
            { return left.global_id < right.global_id; });
  return true;
}


} // namespace dg::hdg::runtime::cpu

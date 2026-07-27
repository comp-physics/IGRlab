#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include "numerics/hdg/core/host_reference.hpp"
#include "numerics/hdg/trace/trace_dg_mpi.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
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
constexpr std::string_view expected_layout =
    "mfem.l2_byNODES+mfem.true_trace_byNODES.p1.n3";

using State = std::array<double, cpu_conserved_components>;
using Matrix4 =
    std::array<std::array<double, cpu_conserved_components>,
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

std::vector<int> BuildPartition(const mfem::Mesh& mesh, const int ranks)
{
  std::vector<int> partition(static_cast<std::size_t>(mesh.GetNE()), 0);
  if (ranks == 1) { return partition; }
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    partition[static_cast<std::size_t>(element)] =
        element * ranks / mesh.GetNE();
  }
  return partition;
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

std::uint64_t SumU64(const std::uint64_t local) noexcept
{
  unsigned long long input = static_cast<unsigned long long>(local);
  unsigned long long output = 0;
  if (MPI_Allreduce(&input, &output, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return 0;
  }
  return static_cast<std::uint64_t>(output);
}

} // namespace

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
                      const std::size_t trace_size)
{
  auto cell = std::make_shared<CpuSlotCell>();
  cell->backing = std::make_unique<CpuSlotStorage>();
  cell->backing->interior.resize(interior_size);
  cell->backing->trace.resize(trace_size);
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
  CpuSlotStorage* storage = Storage(candidate);
  if (!storage) { return; }
  const double invalid = std::numeric_limits<double>::quiet_NaN();
  std::fill(storage->interior.begin(), storage->interior.end(), invalid);
  std::fill(storage->trace.begin(), storage->trace.end(), invalid);
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

bool CpuStateStoragePolicy::Disjoint(
    const CpuStateSlot& left, const CpuStateSlot& right) noexcept
{
  return ValidSlot(left) && ValidSlot(right) &&
      left.interior.cell != right.interior.cell &&
      left.interior.cell->backing.get() != right.interior.cell->backing.get() &&
      left.interior.cell->backing->interior.data() !=
          right.interior.cell->backing->interior.data() &&
      left.interior.cell->backing->trace.data() !=
          right.interior.cell->backing->trace.data();
}

StateHash CpuStateStoragePolicy::HashInterior(
    const CpuInteriorState& state) noexcept
{
  if (!state.cell || !state.cell->backing) { return {}; }
  return HashDoubles(state.cell->backing->interior);
}

StateHash CpuStateStoragePolicy::HashTrace(const CpuTraceState& state) noexcept
{
  if (!state.cell || !state.cell->backing) { return {}; }
  return HashDoubles(state.cell->backing->trace);
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
  const bool reductions =
      MPI_Allreduce(&local_prepared, &all_prepared, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) == MPI_SUCCESS &&
      MPI_Allreduce(&local_mask, &global_mask, 1, MPI_UNSIGNED_LONG_LONG,
                    MPI_BOR, MPI_COMM_WORLD) == MPI_SUCCESS;
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
  const bool reductions =
      MPI_Allreduce(local.data(), reduced.data(), static_cast<int>(local.size()),
                    MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS &&
      MPI_Allreduce(&local_mask, &global_mask, 1, MPI_UNSIGNED_LONG_LONG,
                    MPI_BOR, MPI_COMM_WORLD) == MPI_SUCCESS;
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
  const bool reductions =
      MPI_Allreduce(&local_failure, &global_failure, 1,
                    MPI_UNSIGNED_LONG_LONG, MPI_BOR,
                    MPI_COMM_WORLD) == MPI_SUCCESS &&
      MPI_Allreduce(&local_compatibility, &global_compatibility, 1,
                    MPI_UNSIGNED_LONG_LONG, MPI_BOR,
                    MPI_COMM_WORLD) == MPI_SUCCESS;
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

  void Allocate(const int batch_count, const int ni, const int nt)
  {
    elements = batch_count;
    interior_dofs = ni;
    trace_dofs = nt;
    const std::size_t batches = static_cast<std::size_t>(elements);
    const std::size_t interior = static_cast<std::size_t>(ni);
    const std::size_t trace = static_cast<std::size_t>(nt);
    residual_interior.resize(batches * interior);
    residual_trace.resize(batches * trace);
    a.resize(batches * interior * interior);
    b.resize(batches * interior * trace);
    c.resize(batches * trace * interior);
    d.resize(batches * trace * trace);
    f.resize(batches * interior);
    g.resize(batches * trace);
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

bool ExactConfig(const CpuPersistentConfig& config) noexcept
{
  return config.experiment.valid() && config.experiment ==
          StableIdentity::Copy(expected_experiment) &&
      config.source.valid() && config.source ==
          StableIdentity::Copy(expected_source) &&
      config.polynomial_order == cpu_polynomial_order &&
      config.elements_per_direction == cpu_elements_per_direction &&
      config.assembly_quadrature_order == cpu_assembly_quadrature_order &&
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
      config.allow_one_retry;
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

int PeriodicIndex(const double coordinate) noexcept
{
  int value = static_cast<int>(std::llround(
      coordinate * cpu_elements_per_direction));
  value %= cpu_elements_per_direction;
  if (value < 0) { value += cpu_elements_per_direction; }
  return value;
}

int CellIndex(const double coordinate) noexcept
{
  double periodic = coordinate - std::floor(coordinate);
  int value = static_cast<int>(std::floor(
      periodic * cpu_elements_per_direction + 64.0 *
          std::numeric_limits<double>::epsilon()));
  return std::max(0, std::min(cpu_elements_per_direction - 1, value));
}

} // namespace

struct detail::CpuNumericsCore::Implementation
{
  CpuPersistentConfig config{};
  RuntimeTupleDescriptor tuple{};
  int rank = -1;
  int ranks = 0;
  bool initialized = false;

  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<mfem::L2_FECollection> interior_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> interior_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
  trace::DGInterfaceTraceExchangePlan exchange_plan;

  CpuStateSlot committed_slot{};
  CpuStateSlot candidate_slot{};
  CpuStateSlot checkpoint_slot{};
  std::vector<double> first_iterate_interior;
  std::vector<double> first_iterate_trace;

  std::unique_ptr<mfem::ParGridFunction> old_interior;
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
  std::vector<int> signed_trace_dofs;
  std::vector<std::uint64_t> interior_global_ids;
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
  std::vector<double> local_global_matrix;
  std::vector<double> global_matrix;
  std::unique_ptr<mfem::HypreParVector> hypre_rhs;
  std::unique_ptr<mfem::HypreParVector> hypre_solution;
  std::unique_ptr<mfem::HypreParVector> hypre_residual;
  std::unique_ptr<mfem::HypreILU> ilu;
  std::unique_ptr<mfem::HypreGMRES> gmres;

  CpuRunEvidence evidence{};
  CpuAttemptEvidence active_evidence{};
  StateCoordinates entry_coordinates{};
  StateHash entry_interior_hash{};
  StateHash entry_trace_hash{};

  bool CollectiveAll(const bool local) const noexcept { return AllRanks(local); }

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
    const bool reduced =
        MPI_Allreduce(local, minimum.data(), 4, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) == MPI_SUCCESS &&
        MPI_Allreduce(local, maximum.data(), 4, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) == MPI_SUCCESS;
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
    ++active_evidence.work.phase_entries;
    const bool local_injected = injection.phase == phase &&
        injection.source_rank == rank;
    const bool admitted = CollectiveAll(!local_injected);
    if (!admitted)
    {
      active_evidence.failure_phase = phase;
      active_evidence.failure_source_rank = injection.source_rank;
    }
    return admitted;
  }

  void CompletePhase() noexcept
  {
    ++active_evidence.work.phase_completions;
  }

  bool ProjectTrace(mfem::ParGridFunction& local_trace,
                    mfem::Vector& true_trace,
                    const bool perturbed) noexcept
  {
    try
    {
      local_trace = 0.0;
      for (int face = 0; face < mesh->GetNumFaces(); ++face)
      {
        const mfem::FiniteElement* finite_element =
            trace_space->GetFaceElement(face);
        mfem::ElementTransformation* transformation =
            mesh->GetFaceTransformation(face);
        if (!finite_element || !transformation) { return false; }
        mfem::Array<int> vdofs;
        trace_space->GetFaceVDofs(face, vdofs);
        const int scalar_dofs = finite_element->GetDof();
        if (vdofs.Size() != scalar_dofs * cpu_conserved_components)
        {
          return false;
        }
        const mfem::IntegrationRule& nodes = finite_element->GetNodes();
        for (int scalar = 0; scalar < scalar_dofs; ++scalar)
        {
          mfem::Vector physical(cpu_dimension);
          transformation->Transform(nodes.IntPoint(scalar), physical);
          State state{};
          const bool ok = perturbed
              ? PerturbedConservative(physical[0], physical[1], state)
              : ManufacturedConservative(physical[0], physical[1], state);
          if (!ok) { return false; }
          for (int component = 0; component < cpu_conserved_components;
               ++component)
          {
            double sign = 1.0;
            const int dof = Decode(
                vdofs[ByNodes(scalar, scalar_dofs, component)], sign);
            if (dof < 0 || dof >= local_trace.Size()) { return false; }
            local_trace[dof] = sign * state[component];
          }
        }
      }
      local_trace.ParallelProject(true_trace);
      return Finite(true_trace);
    }
    catch (...)
    {
      return false;
    }
  }

  bool LoadTrace(const mfem::Vector& true_values,
                 mfem::ParGridFunction& local_values) noexcept
  {
    try
    {
      trace_distribution_workspace = 0.0;
      if (!exchange_plan.Distribute(*trace_space, true_values,
                                    trace_distribution_workspace))
      {
        return false;
      }
      const double* source = trace_distribution_workspace.HostRead();
      double* destination = local_values.HostWrite();
      if ((!source || !destination) &&
          trace_distribution_workspace.Size() != 0)
      {
        return false;
      }
      std::memcpy(destination, source,
                  static_cast<std::size_t>(
                      trace_distribution_workspace.Size()) *
                      sizeof(double));
      return true;
    }
    catch (...)
    {
      return false;
    }
  }

  bool BuildCanonicalIds() noexcept
  {
    try
    {
      interior_global_ids.resize(
          static_cast<std::size_t>(interior_space->GetVSize()));
      for (int local = 0; local < interior_space->GetVSize(); ++local)
      {
        interior_global_ids[static_cast<std::size_t>(local)] =
            static_cast<std::uint64_t>(
                interior_space->GetGlobalTDofNumber(local));
      }
      trace_canonical_ids.assign(
          static_cast<std::size_t>(trace_space->GetTrueVSize()),
          std::numeric_limits<std::uint64_t>::max());
      for (int face = 0; face < mesh->GetNumFaces(); ++face)
      {
        const mfem::FiniteElement* finite_element =
            trace_space->GetFaceElement(face);
        mfem::ElementTransformation* transformation =
            mesh->GetFaceTransformation(face);
        if (!finite_element || !transformation || finite_element->GetDof() != 2)
        {
          return false;
        }
        mfem::Array<int> vdofs;
        trace_space->GetFaceVDofs(face, vdofs);
        const mfem::IntegrationRule& nodes = finite_element->GetNodes();
        std::array<mfem::Vector, 2> physical{{mfem::Vector(cpu_dimension),
                                              mfem::Vector(cpu_dimension)}};
        transformation->Transform(nodes.IntPoint(0), physical[0]);
        transformation->Transform(nodes.IntPoint(1), physical[1]);
        const bool vertical =
            std::abs(physical[0][0] - physical[1][0]) < 1.0e-12;
        const double fixed = vertical
            ? 0.5 * (physical[0][0] + physical[1][0])
            : 0.5 * (physical[0][1] + physical[1][1]);
        const double varying0 = vertical ? physical[0][1] : physical[0][0];
        const double varying1 = vertical ? physical[1][1] : physical[1][0];
        const int fixed_index = PeriodicIndex(fixed);
        const int cell_index = CellIndex(std::min(varying0, varying1));
        const int face_index = vertical
            ? fixed_index * cpu_elements_per_direction + cell_index
            : cpu_elements_per_direction * cpu_elements_per_direction +
                fixed_index * cpu_elements_per_direction + cell_index;
        for (int scalar = 0; scalar < 2; ++scalar)
        {
          const int canonical_scalar =
              (scalar == 0 ? varying0 : varying1) <=
                      (scalar == 0 ? varying1 : varying0)
                  ? 0 : 1;
          for (int component = 0; component < cpu_conserved_components;
               ++component)
          {
            double sign = 1.0;
            const int local_dof = Decode(
                vdofs[ByNodes(scalar, 2, component)], sign);
            if (local_dof < 0 ||
                local_dof >= static_cast<int>(exchange_plan.map.dofs.entries.size()))
            {
              return false;
            }
            const trace::TraceMapEntry& entry =
                exchange_plan.map.dofs.entries[
                    static_cast<std::size_t>(local_dof)];
            if (entry.ownership != Ownership::owned) { continue; }
            const std::uint64_t canonical =
                (static_cast<std::uint64_t>(face_index * 2 + canonical_scalar) *
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

  bool BuildTopology() noexcept
  {
    try
    {
      const mfem::FiniteElement* interior = interior_space->GetFE(0);
      if (!interior || interior->GetDof() != 4 || mesh->GetNE() <= 0)
      {
        return false;
      }
      scalar_interior_dofs = interior->GetDof();
      local_interior = scalar_interior_dofs * cpu_conserved_components;
      mfem::Array<int> first_faces, first_orientations;
      mesh->GetElementEdges(0, first_faces, first_orientations);
      if (first_faces.Size() != 4) { return false; }
      scalar_trace_dofs = 2;
      local_trace = first_faces.Size() * scalar_trace_dofs *
          cpu_conserved_components;
      element_interior_vdofs.resize(
          static_cast<std::size_t>(mesh->GetNE()) * local_interior);
      element_faces.resize(static_cast<std::size_t>(mesh->GetNE()) * 4);
      signed_trace_dofs.resize(
          static_cast<std::size_t>(mesh->GetNE()) * local_trace);
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
              scalar_trace_dofs * cpu_conserved_components)
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
        local_interior / cpu_conserved_components,
        local_trace / cpu_conserved_components,
        cpu_conserved_components, cpu_conserved_components};
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
    true_matrix->HostReadWrite();
    true_matrix->GetDiag(true_diagonal_view);
    true_matrix->GetOffd(true_offdiagonal_view,
                         true_offdiagonal_columns);
    const HYPRE_BigInt* row_starts = true_matrix->GetRowStarts();
    int* diagonal_i = true_diagonal_view.GetI();
    int* diagonal_j = true_diagonal_view.GetJ();
    double* diagonal_data = true_diagonal_view.GetData();
    double* offdiagonal_data = true_offdiagonal_view.GetData();
    if (!row_starts || !diagonal_i || !diagonal_j || !diagonal_data ||
        (true_offdiagonal_view.NumNonZeroElems() != 0 &&
         !offdiagonal_data))
    {
      return false;
    }
    const HYPRE_BigInt row_begin = HYPRE_AssumedPartitionCheck()
        ? row_starts[0] : row_starts[rank];
    std::fill(diagonal_data,
              diagonal_data + true_diagonal_view.NumNonZeroElems(), 0.0);
    if (true_offdiagonal_view.NumNonZeroElems() != 0)
    {
      std::fill(offdiagonal_data,
                offdiagonal_data +
                    true_offdiagonal_view.NumNonZeroElems(), 0.0);
    }
    for (int row = 0; row < true_diagonal_view.Height(); ++row)
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
    true_matrix->HypreRead();
    *hypre_rhs = 1.0;
    *hypre_solution = 0.0;
    gmres->Mult(*hypre_rhs, *hypre_solution);
    *hypre_rhs = 0.0;
    *hypre_solution = 0.0;
    *hypre_residual = 0.0;
    return true;
  }

  bool AllocateWorkspace() noexcept
  {
    try
    {
      current_system.Allocate(mesh->GetNE(), local_interior, local_trace);
      trial_system.Allocate(mesh->GetNE(), local_interior, local_trace);
      const std::size_t batches = static_cast<std::size_t>(mesh->GetNE());
      const std::size_t ni = static_cast<std::size_t>(local_interior);
      const std::size_t nt = static_cast<std::size_t>(local_trace);
      schur.resize(batches * nt * nt);
      reduced_rhs.resize(batches * nt);
      local_trace_increment.resize(batches * nt);
      recovered_interior.resize(batches * ni);
      grid_interior_increment.resize(
          static_cast<std::size_t>(interior_space->GetVSize()));
      interior_census.resize(grid_interior_increment.size());

      old_interior = std::make_unique<mfem::ParGridFunction>(
          interior_space.get());
      iterate_interior = std::make_unique<mfem::ParGridFunction>(
          interior_space.get());
      trial_interior = std::make_unique<mfem::ParGridFunction>(
          interior_space.get());
      iterate_trace = std::make_unique<mfem::ParGridFunction>(trace_space.get());
      trial_trace = std::make_unique<mfem::ParGridFunction>(trace_space.get());
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
      true_matrix.reset(parallel_form->ParallelAssemble(local_matrix.get()));
      if (!true_matrix ||
          true_matrix->Height() != trace_space->GetTrueVSize() ||
          true_matrix->GetGlobalNumRows() != trace_space->GlobalTrueVSize() ||
          true_matrix->GetGlobalNumCols() != trace_space->GlobalTrueVSize())
      {
        return false;
      }
      true_matrix->GetDiag(true_diagonal_view);
      true_matrix->GetOffd(true_offdiagonal_view,
                           true_offdiagonal_columns);
      const std::size_t global_trace = static_cast<std::size_t>(
          trace_space->GlobalTrueVSize());
      local_global_matrix.resize(global_trace * global_trace);
      global_matrix.resize(global_trace * global_trace);
      *local_matrix = 0.0;
      if (!PrimeCondensationWorkspace()) { return false; }
      hypre_rhs = std::make_unique<mfem::HypreParVector>(trace_space.get());
      hypre_solution =
          std::make_unique<mfem::HypreParVector>(trace_space.get());
      hypre_residual =
          std::make_unique<mfem::HypreParVector>(trace_space.get());
      ilu = std::make_unique<mfem::HypreILU>();
      ilu->SetType(0);
      ilu->SetLevelOfFill(0);
      ilu->SetMaxIter(1);
      ilu->SetTol(0.0);
      ilu->SetLocalReordering(0);
      ilu->SetPrintLevel(0);
      gmres = std::make_unique<mfem::HypreGMRES>(MPI_COMM_WORLD);
      gmres->SetTol(1.0e-12);
      gmres->SetAbsTol(0.0);
      const int global_trace_size =
          static_cast<int>(trace_space->GlobalTrueVSize());
      gmres->SetMaxIter(std::max(100, global_trace_size));
      gmres->SetKDim(std::min(100, std::max(20, global_trace_size)));
      gmres->SetLogging(1);
      gmres->SetPrintLevel(0);
      gmres->SetPreconditioner(*ilu);
      gmres->SetOperator(*true_matrix);
      return PrimeHypreWorkspace();
    }
    catch (...)
    {
      return false;
    }
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
        bytes(local_global_matrix) + bytes(global_matrix);
  }

  bool Assemble(const mfem::ParGridFunction& current_interior,
                const mfem::ParGridFunction& current_trace,
                LocalSystem& system) noexcept
  {
    system.Zero();
    try
    {
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
            finite_element->GetGeomType(), config.assembly_quadrature_order);
        const std::size_t element_ni =
            static_cast<std::size_t>(element) * local_interior;
        const std::size_t element_a = element_ni * local_interior;
        for (int q = 0; q < volume_rule.GetNPoints(); ++q)
        {
          const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
          transformation->SetIntPoint(&point);
          transformation->Transform(point, physical_workspace);
          finite_element->CalcShape(point, volume_shape_workspace);
          finite_element->CalcPhysDShape(*transformation,
                                         volume_derivative_workspace);
          State state{}, old_state{}, source{}, flux_x{}, flux_y{};
          Matrix4 jacobian_x{}, jacobian_y{};
          if (!Interpolate(current_interior, interior_vdofs,
                           scalar_interior_dofs, volume_shape_workspace,
                           state) ||
              !Interpolate(*old_interior, interior_vdofs,
                           scalar_interior_dofs, volume_shape_workspace,
                           old_state) ||
              !ManufacturedSource(physical_workspace[0],
                                  physical_workspace[1], source) ||
              !NormalFlux(state, {{1.0, 0.0}}, flux_x) ||
              !NormalFlux(state, {{0.0, 1.0}}, flux_y) ||
              !NormalFluxJacobian(state, {{1.0, 0.0}}, jacobian_x) ||
              !NormalFluxJacobian(state, {{0.0, 1.0}}, jacobian_y))
          {
            return false;
          }
          const double weight = point.weight * transformation->Weight();
          if (!Finite(weight) || !(weight > 0.0)) { return false; }
          for (int row = 0; row < scalar_interior_dofs; ++row)
          {
            for (int row_component = 0;
                 row_component < cpu_conserved_components; ++row_component)
            {
              const int local_row =
                  ByNodes(row, scalar_interior_dofs, row_component);
              system.residual_interior[element_ni + local_row] += weight *
                  (volume_shape_workspace[row] *
                       (state[row_component] - old_state[row_component]) /
                       config.dt -
                   volume_derivative_workspace(row, 0) *
                       flux_x[row_component] -
                   volume_derivative_workspace(row, 1) *
                       flux_y[row_component] -
                   volume_shape_workspace[row] * source[row_component]);
            }
            for (int column = 0; column < scalar_interior_dofs; ++column)
              for (int row_component = 0;
                   row_component < cpu_conserved_components; ++row_component)
                for (int column_component = 0;
                     column_component < cpu_conserved_components;
                     ++column_component)
                {
                  const int local_row =
                      ByNodes(row, scalar_interior_dofs, row_component);
                  const int local_column = ByNodes(
                      column, scalar_interior_dofs, column_component);
                  const double mass = row_component == column_component
                      ? volume_shape_workspace[row] *
                            volume_shape_workspace[column] / config.dt
                      : 0.0;
                  const double volume =
                      -(volume_derivative_workspace(row, 0) *
                            jacobian_x[row_component][column_component] +
                        volume_derivative_workspace(row, 1) *
                            jacobian_y[row_component][column_component]) *
                      volume_shape_workspace[column];
                  system.a[element_a +
                           static_cast<std::size_t>(local_row) *
                               local_interior +
                           local_column] += weight * (mass + volume);
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
          const bool element_one = face_transformation->Elem1No == element;
          if (!element_one && face_transformation->Elem2No != element)
          {
            return false;
          }
          const int* face_trace_vdofs = signed_trace_dofs.data() +
              static_cast<std::size_t>(element) * local_trace + trace_offset;
          const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
              trace_element->GetGeomType(), config.assembly_quadrature_order);
          for (int q = 0; q < face_rule.GetNPoints(); ++q)
          {
            const mfem::IntegrationPoint& face_point = face_rule.IntPoint(q);
            face_transformation->SetAllIntPoints(&face_point);
            const mfem::IntegrationPoint& element_point = element_one
                ? face_transformation->GetElement1IntPoint()
                : face_transformation->GetElement2IntPoint();
            mfem::CalcOrtho(face_transformation->Face->Jacobian(),
                            scaled_normal_workspace);
            const double normal_norm = scaled_normal_workspace.Norml2();
            const double face_measure = face_transformation->Face->Weight();
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
            State interior_state{}, trace_state{}, trace_flux{};
            Matrix4 trace_jacobian{};
            if (!Interpolate(current_interior, interior_vdofs,
                             scalar_interior_dofs,
                             face_interior_shape_workspace,
                             interior_state) ||
                !Interpolate(current_trace, face_trace_vdofs,
                             scalar_trace_dofs, face_trace_shape_workspace,
                             trace_state) ||
                !NormalFlux(trace_state, normal, trace_flux) ||
                !NormalFluxJacobian(trace_state, normal, trace_jacobian))
            {
              return false;
            }
            const double weight = face_point.weight * face_measure;
            for (int row = 0; row < scalar_interior_dofs; ++row)
              for (int component = 0; component < cpu_conserved_components;
                   ++component)
              {
                const int local_row =
                    ByNodes(row, scalar_interior_dofs, component);
                system.residual_interior[element_ni + local_row] +=
                    weight * face_interior_shape_workspace[row] *
                    (trace_flux[component] + cpu_fixed_tau *
                         (interior_state[component] - trace_state[component]));
              }
            for (int row = 0; row < scalar_trace_dofs; ++row)
              for (int component = 0; component < cpu_conserved_components;
                   ++component)
              {
                const int local_row = trace_offset +
                    ByNodes(row, scalar_trace_dofs, component);
                system.residual_trace[
                    static_cast<std::size_t>(element) * local_trace +
                    local_row] += weight * face_trace_shape_workspace[row] *
                    (trace_flux[component] + cpu_fixed_tau *
                         (interior_state[component] - trace_state[component]));
              }
            for (int row = 0; row < scalar_interior_dofs; ++row)
              for (int column = 0; column < scalar_interior_dofs; ++column)
                for (int component = 0;
                     component < cpu_conserved_components; ++component)
                {
                  const int local_row =
                      ByNodes(row, scalar_interior_dofs, component);
                  const int local_column =
                      ByNodes(column, scalar_interior_dofs, component);
                  system.a[element_a +
                           static_cast<std::size_t>(local_row) *
                               local_interior +
                               local_column] +=
                      weight * face_interior_shape_workspace[row] *
                      cpu_fixed_tau * face_interior_shape_workspace[column];
                }
            for (int row = 0; row < scalar_interior_dofs; ++row)
              for (int column = 0; column < scalar_trace_dofs; ++column)
                for (int row_component = 0;
                     row_component < cpu_conserved_components;
                     ++row_component)
                  for (int column_component = 0;
                       column_component < cpu_conserved_components;
                       ++column_component)
                  {
                    const int local_row =
                        ByNodes(row, scalar_interior_dofs, row_component);
                    const int local_column = trace_offset +
                        ByNodes(column, scalar_trace_dofs, column_component);
                    system.b[static_cast<std::size_t>(element) *
                                 local_interior * local_trace +
                             static_cast<std::size_t>(local_row) *
                                 local_trace +
                             local_column] +=
                        weight * face_interior_shape_workspace[row] *
                        face_trace_shape_workspace[column] *
                        (trace_jacobian[row_component][column_component] -
                         (row_component == column_component
                              ? cpu_fixed_tau : 0.0));
                  }
            for (int row = 0; row < scalar_trace_dofs; ++row)
              for (int column = 0; column < scalar_interior_dofs; ++column)
                for (int component = 0;
                     component < cpu_conserved_components; ++component)
                {
                  const int local_row = trace_offset +
                      ByNodes(row, scalar_trace_dofs, component);
                  const int local_column =
                      ByNodes(column, scalar_interior_dofs, component);
                  system.c[static_cast<std::size_t>(element) * local_trace *
                               local_interior +
                           static_cast<std::size_t>(local_row) *
                               local_interior +
                           local_column] +=
                      weight * face_trace_shape_workspace[row] *
                      cpu_fixed_tau * face_interior_shape_workspace[column];
                }
            for (int row = 0; row < scalar_trace_dofs; ++row)
              for (int column = 0; column < scalar_trace_dofs; ++column)
                for (int row_component = 0;
                     row_component < cpu_conserved_components;
                     ++row_component)
                  for (int column_component = 0;
                       column_component < cpu_conserved_components;
                       ++column_component)
                  {
                    const int local_row = trace_offset +
                        ByNodes(row, scalar_trace_dofs, row_component);
                    const int local_column = trace_offset +
                        ByNodes(column, scalar_trace_dofs, column_component);
                    system.d[static_cast<std::size_t>(element) * local_trace *
                                 local_trace +
                             static_cast<std::size_t>(local_row) *
                                 local_trace +
                                 local_column] +=
                        weight * face_trace_shape_workspace[row] *
                        face_trace_shape_workspace[column] *
                        (trace_jacobian[row_component][column_component] -
                         (row_component == column_component
                              ? cpu_fixed_tau : 0.0));
                  }
          }
          trace_offset += scalar_trace_dofs * cpu_conserved_components;
        }
        if (trace_offset != local_trace) { return false; }
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

  double ResidualNorm(const LocalSystem& system) noexcept
  {
    try
    {
      local_trace_residual = 0.0;
      for (int element = 0; element < system.elements; ++element)
        for (int local = 0; local < system.trace_dofs; ++local)
        {
          double sign = 1.0;
          const int dof = Decode(signed_trace_dofs[
              static_cast<std::size_t>(element) * system.trace_dofs + local],
                                 sign);
          if (dof < 0 || dof >= local_trace_residual.Size())
          {
            return std::numeric_limits<double>::quiet_NaN();
          }
          local_trace_residual[dof] += sign * system.residual_trace[
              static_cast<std::size_t>(element) * system.trace_dofs + local];
        }
      trace_space->Dof_TrueDof_Matrix()->MultTranspose(
          local_trace_residual, true_trace_residual);
      double local_interior_squared = 0.0;
      for (const double value : system.residual_interior)
      {
        local_interior_squared += value * value;
      }
      double global_interior_squared = 0.0;
      if (MPI_Allreduce(&local_interior_squared, &global_interior_squared, 1,
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
      {
        return std::numeric_limits<double>::quiet_NaN();
      }
      const double trace_squared = mfem::InnerProduct(
          MPI_COMM_WORLD, true_trace_residual, true_trace_residual);
      return std::sqrt(global_interior_squared + trace_squared);
    }
    catch (...)
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
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
    return primitive[0] >= density_gate && primitive[3] >= pressure_gate &&
        wave_speed < cpu_fixed_tau;
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
              local_face * scalar_trace_dofs * cpu_conserved_components;
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
    if (MPI_Allreduce(&valid, &global_valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS)
    {
      return false;
    }
    MPI_Allreduce(&local_density, &minimum_density, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_pressure, &minimum_pressure, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_wave, &maximum_wave_speed, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    return global_valid == 1 && Finite(minimum_density) &&
        Finite(minimum_pressure) && Finite(maximum_wave_speed) &&
        minimum_density >= density_gate && minimum_pressure >= pressure_gate &&
        maximum_wave_speed < cpu_fixed_tau;
  }

  bool CondenseAndAssemble(const LocalSystem& system,
                           StateHash& matrix_hash) noexcept
  {
    try
    {
      const LocalBlockShape shape{
          local_interior / cpu_conserved_components,
          local_trace / cpu_conserved_components,
          cpu_conserved_components, cpu_conserved_components};
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
                static_cast<std::size_t>(element) * local_trace * local_trace +
                static_cast<std::size_t>(row) * local_trace + column];
          }
        }
        local_matrix->AddSubMatrix(element_trace_dofs_workspace,
                                   element_trace_dofs_workspace,
                                   local_schur_workspace, 1);
      }
      const int global_trace =
          static_cast<int>(trace_space->GlobalTrueVSize());
      if (!true_matrix || global_trace <= 0 ||
          local_global_matrix.size() !=
              static_cast<std::size_t>(global_trace) * global_trace ||
          global_matrix.size() != local_global_matrix.size())
      {
        return false;
      }
      std::fill(local_global_matrix.begin(), local_global_matrix.end(), 0.0);
      for (int element = 0; element < system.elements; ++element)
        for (int row = 0; row < local_trace; ++row)
        {
          double row_sign = 1.0;
          const int row_dof = Decode(signed_trace_dofs[
              static_cast<std::size_t>(element) * local_trace + row],
                                     row_sign);
          if (row_dof < 0 || row_dof >=
                  static_cast<int>(exchange_plan.map.dofs.entries.size()))
          {
            return false;
          }
          const std::uint64_t global_row =
              exchange_plan.map.dofs.entries[
                  static_cast<std::size_t>(row_dof)].global_true_dof;
          for (int column = 0; column < local_trace; ++column)
          {
            double column_sign = 1.0;
            const int column_dof = Decode(signed_trace_dofs[
                static_cast<std::size_t>(element) * local_trace + column],
                                          column_sign);
            if (column_dof < 0 || column_dof >= static_cast<int>(
                    exchange_plan.map.dofs.entries.size()))
            {
              return false;
            }
            const std::uint64_t global_column =
                exchange_plan.map.dofs.entries[
                    static_cast<std::size_t>(column_dof)].global_true_dof;
            local_global_matrix[
                static_cast<std::size_t>(global_row) * global_trace +
                global_column] += row_sign * column_sign * schur[
                    static_cast<std::size_t>(element) * local_trace *
                        local_trace +
                    static_cast<std::size_t>(row) * local_trace + column];
          }
        }
      if (local_global_matrix.size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          MPI_Allreduce(local_global_matrix.data(), global_matrix.data(),
                        static_cast<int>(local_global_matrix.size()),
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS ||
          !Finite(global_matrix))
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
      int* offdiagonal_i = true_offdiagonal_view.GetI();
      int* offdiagonal_j = true_offdiagonal_view.GetJ();
      double* offdiagonal_data = true_offdiagonal_view.GetData();
      if (!row_starts || !diagonal_i || !diagonal_j || !diagonal_data ||
          !offdiagonal_i ||
          (true_offdiagonal_view.NumNonZeroElems() != 0 &&
           (!offdiagonal_j || !offdiagonal_data ||
            !true_offdiagonal_columns)))
      {
        return false;
      }
      const HYPRE_BigInt row_begin = HYPRE_AssumedPartitionCheck()
          ? row_starts[0] : row_starts[rank];
      for (int row = 0; row < true_diagonal_view.Height(); ++row)
      {
        const std::size_t global_row =
            static_cast<std::size_t>(row_begin + row);
        for (int entry = diagonal_i[row]; entry < diagonal_i[row + 1];
             ++entry)
        {
          const std::size_t global_column =
              static_cast<std::size_t>(row_begin + diagonal_j[entry]);
          diagonal_data[entry] =
              global_matrix[global_row * global_trace + global_column];
        }
        for (int entry = offdiagonal_i[row];
             entry < offdiagonal_i[row + 1]; ++entry)
        {
          const std::size_t global_column = static_cast<std::size_t>(
              true_offdiagonal_columns[offdiagonal_j[entry]]);
          offdiagonal_data[entry] =
              global_matrix[global_row * global_trace + global_column];
        }
      }
      true_matrix->HypreRead();
      trace_space->Dof_TrueDof_Matrix()->MultTranspose(
          local_trace_rhs, true_trace_rhs);
      matrix_hash = HashPair(HashSystemMatrix(system), HashDoubles(schur));
      return Finite(true_trace_rhs);
    }
    catch (...)
    {
      return false;
    }
  }

  bool SolveTrace(double& true_relative_residual, int& iterations,
                  StateHash& factor_hash,
                  StateHash& solution_hash) noexcept
  {
    try
    {
      const double* rhs_source = true_trace_rhs.HostRead();
      double* rhs_destination = hypre_rhs->HostWrite();
      if ((!rhs_source || !rhs_destination) && true_trace_rhs.Size() != 0)
      {
        return false;
      }
      std::memcpy(rhs_destination, rhs_source,
                  static_cast<std::size_t>(true_trace_rhs.Size()) *
                      sizeof(double));
      *hypre_solution = 0.0;
      const std::array<char, 20> factor_identity{{
          'h','y','p','r','e','-','i','l','u','0','-','t','y','p','e','0',
          '-','p','1','\0'}};
      factor_hash = HashPair(HashDoubles(schur),
                             HashBytes(factor_identity.data(),
                                       factor_identity.size()));
      gmres->Mult(*hypre_rhs, *hypre_solution);
      gmres->GetNumIterations(iterations);
      if (iterations <= 0) { return false; }
      true_matrix->Mult(*hypre_solution, *hypre_residual);
      *hypre_residual -= *hypre_rhs;
      const double residual_squared = mfem::InnerProduct(
          MPI_COMM_WORLD, *hypre_residual, *hypre_residual);
      const double rhs_squared = mfem::InnerProduct(
          MPI_COMM_WORLD, *hypre_rhs, *hypre_rhs);
      true_relative_residual = std::sqrt(residual_squared) /
          std::max(std::sqrt(rhs_squared), 1.0e-300);
      const double* solution = hypre_solution->HostRead();
      double* increment = true_increment.HostWrite();
      if ((!solution || !increment) && true_increment.Size() != 0)
      {
        return false;
      }
      std::memcpy(increment, solution,
                  static_cast<std::size_t>(true_increment.Size()) *
                      sizeof(double));
      solution_hash = HashBytes(solution,
                                static_cast<std::size_t>(true_increment.Size()) *
                                    sizeof(double));
      return Finite(true_relative_residual) &&
          true_relative_residual <= config.true_residual_tolerance;
    }
    catch (...)
    {
      return false;
    }
  }

  bool Recover() noexcept
  {
    try
    {
      distributed_increment = 0.0;
      if (!exchange_plan.Distribute(*trace_space, true_increment,
                                    distributed_increment))
      {
        return false;
      }
      const double* distributed = distributed_increment.HostRead();
      if (!distributed) { return false; }
      for (int element = 0; element < mesh->GetNE(); ++element)
        for (int local = 0; local < local_trace; ++local)
        {
          double sign = 1.0;
          const int dof = Decode(signed_trace_dofs[
              static_cast<std::size_t>(element) * local_trace + local], sign);
          if (dof < 0 || dof >= distributed_increment.Size()) { return false; }
          local_trace_increment[
              static_cast<std::size_t>(element) * local_trace + local] =
              sign * distributed[dof];
        }
      if (condensation.Recover(
              AsConst(View(local_trace_increment.data(), mesh->GetNE(),
                           local_trace, 1)),
              View(recovered_interior.data(), mesh->GetNE(), local_interior,
                   1)) != core::Status::success ||
          !Finite(recovered_interior))
      {
        return false;
      }
      std::fill(grid_interior_increment.begin(),
                grid_interior_increment.end(), 0.0);
      std::fill(interior_census.begin(), interior_census.end(), 0);
      for (int element = 0; element < mesh->GetNE(); ++element)
        for (int local = 0; local < local_interior; ++local)
        {
          double sign = 1.0;
          const int dof = Decode(element_interior_vdofs[
              static_cast<std::size_t>(element) * local_interior + local],
                                 sign);
          if (dof < 0 ||
              dof >= static_cast<int>(grid_interior_increment.size()))
          {
            return false;
          }
          grid_interior_increment[static_cast<std::size_t>(dof)] = sign *
              recovered_interior[
                  static_cast<std::size_t>(element) * local_interior + local];
          ++interior_census[static_cast<std::size_t>(dof)];
        }
      return std::all_of(interior_census.begin(), interior_census.end(),
                         [](const int count) { return count == 1; }) &&
          Finite(grid_interior_increment);
    }
    catch (...)
    {
      return false;
    }
  }

  bool MakeTrial(const double alpha) noexcept
  {
    try
    {
      const double* interior = iterate_interior->HostRead();
      double* trial = trial_interior->HostWrite();
      const double* trace_values = iterate_true_trace.HostRead();
      const double* trace_delta = true_increment.HostRead();
      double* trace_trial = trial_true_trace.HostWrite();
      if (!interior || !trial || !trace_values || !trace_delta || !trace_trial)
      {
        return false;
      }
      for (int index = 0; index < iterate_interior->Size(); ++index)
      {
        trial[index] = interior[index] + alpha *
            grid_interior_increment[static_cast<std::size_t>(index)];
      }
      for (int index = 0; index < iterate_true_trace.Size(); ++index)
      {
        trace_trial[index] = trace_values[index] + alpha * trace_delta[index];
      }
      return LoadTrace(trial_true_trace, *trial_trace);
    }
    catch (...)
    {
      return false;
    }
  }

  bool AcceptTrial() noexcept
  {
    try
    {
      const double* interior = trial_interior->HostRead();
      double* next_interior = iterate_interior->HostWrite();
      const double* trace_values = trial_true_trace.HostRead();
      double* next_trace = iterate_true_trace.HostWrite();
      if (!interior || !next_interior || !trace_values || !next_trace)
      {
        return false;
      }
      std::memcpy(next_interior, interior,
                  static_cast<std::size_t>(iterate_interior->Size()) *
                      sizeof(double));
      std::memcpy(next_trace, trace_values,
                  static_cast<std::size_t>(iterate_true_trace.Size()) *
                      sizeof(double));
      if (!LoadTrace(iterate_true_trace, *iterate_trace)) { return false; }
      CopySystem(trial_system, current_system);
      return true;
    }
    catch (...)
    {
      return false;
    }
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

StateHash CollectiveHash(const StateHash& local, const int ranks) noexcept
{
  std::array<StateHash, 2> hashes{};
  if (MPI_Allgather(local.bytes.data(), static_cast<int>(local.bytes.size()),
                    MPI_BYTE, hashes.data(),
                    static_cast<int>(local.bytes.size()), MPI_BYTE,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
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
    : implementation_(std::make_unique<Implementation>())
{}

detail::CpuNumericsCore::~CpuNumericsCore() = default;
detail::CpuNumericsCore::CpuNumericsCore(CpuNumericsCore&&) noexcept = default;
detail::CpuNumericsCore& detail::CpuNumericsCore::operator=(
    CpuNumericsCore&&) noexcept = default;

bool detail::CpuNumericsCore::Initialize(
    const CpuPersistentConfig& config,
    const RuntimeTupleDescriptor& tuple,
    CpuInitialization& initialized) noexcept
{
  initialized = {};
  Implementation& state = *implementation_;
  if (state.initialized) { return false; }
  int mpi_initialized = 0;
  int mpi_finalized = 0;
  MPI_Initialized(&mpi_initialized);
  MPI_Finalized(&mpi_finalized);
  bool local_ok = mpi_initialized != 0 && mpi_finalized == 0 &&
      MPI_Comm_rank(MPI_COMM_WORLD, &state.rank) == MPI_SUCCESS &&
      MPI_Comm_size(MPI_COMM_WORLD, &state.ranks) == MPI_SUCCESS &&
      (state.ranks == 1 || state.ranks == 2) && ExactConfig(config) &&
      ExactTuple(tuple, state.ranks);
  if (!AllRanks(local_ok)) { return false; }
  state.config = config;
  state.tuple = tuple;

  std::vector<int> partition;
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        cpu_elements_per_direction, cpu_elements_per_direction,
        mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(cpu_dimension));
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations));
    partition = BuildPartition(periodic, state.ranks);
    state.mesh = std::make_unique<mfem::ParMesh>(
        MPI_COMM_WORLD, periodic, partition.data());
    state.interior_collection = std::make_unique<mfem::L2_FECollection>(
        cpu_polynomial_order, cpu_dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    state.trace_collection =
        std::make_unique<mfem::DG_Interface_FECollection>(
            cpu_polynomial_order, cpu_dimension, mfem::FiniteElement::VALUE,
            mfem::BasisType::GaussLegendre);
    state.interior_space = std::make_unique<mfem::ParFiniteElementSpace>(
        state.mesh.get(), state.interior_collection.get(),
        cpu_conserved_components, mfem::Ordering::byNODES);
    state.trace_space = std::make_unique<mfem::ParFiniteElementSpace>(
        state.mesh.get(), state.trace_collection.get(),
        cpu_conserved_components, mfem::Ordering::byNODES);
    state.interior_space->ExchangeFaceNbrData();
    state.trace_space->ExchangeFaceNbrData();
    local_ok = state.mesh->GetNE() > 0 &&
        state.interior_space->GetVSize() > 0 &&
        state.trace_space->GetVSize() > 0 &&
        state.trace_space->GetTrueVSize() > 0 &&
        trace::BuildDGInterfaceTraceExchangePlan(
            *state.trace_space, state.exchange_plan) &&
        state.exchange_plan.IsConsistent();
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!AllRanks(local_ok)) { return false; }
  local_ok = state.BuildTopology();
  if (!AllRanks(local_ok)) { return false; }

  try
  {
    const std::size_t interior_size =
        static_cast<std::size_t>(state.interior_space->GetVSize());
    const std::size_t trace_size =
        static_cast<std::size_t>(state.trace_space->GetTrueVSize());
    state.committed_slot = MakeSlot(interior_size, trace_size);
    state.candidate_slot = MakeSlot(interior_size, trace_size);
    state.checkpoint_slot = MakeSlot(interior_size, trace_size);
    state.first_iterate_interior.resize(interior_size);
    state.first_iterate_trace.resize(trace_size);
    local_ok = state.AllocateWorkspace();
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!AllRanks(local_ok)) { return false; }

  try
  {
    auto exact_function = [](const mfem::Vector& point, mfem::Vector& value)
    {
      State conservative{};
      if (!ManufacturedConservative(point[0], point[1], conservative))
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
    auto perturbed_function = [](const mfem::Vector& point,
                                 mfem::Vector& value)
    {
      State conservative{};
      if (!PerturbedConservative(point[0], point[1], conservative))
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
    local_ok = Copy(*state.old_interior,
                    state.Committed()->interior) &&
        Copy(*state.trial_interior, state.first_iterate_interior) &&
        state.ProjectTrace(*state.iterate_trace, state.iterate_true_trace,
                           false) &&
        state.LoadTrace(state.iterate_true_trace, *state.iterate_trace) &&
        Copy(state.iterate_true_trace, state.Committed()->trace) &&
        state.ProjectTrace(*state.trial_trace, state.trial_true_trace, true) &&
        state.LoadTrace(state.trial_true_trace, *state.trial_trace) &&
        Copy(state.trial_true_trace, state.first_iterate_trace) &&
        CopySlot(state.committed_slot, state.candidate_slot) &&
        CopySlot(state.committed_slot, state.checkpoint_slot);
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!AllRanks(local_ok)) { return false; }

  const StateHash local_mesh = HashBytes(
      &cpu_elements_per_direction, sizeof(cpu_elements_per_direction));
  const StateHash local_partition = HashTopologyIntegers(partition);
  const StateHash local_topology = HashPair(
      HashTopologyIntegers(state.signed_trace_dofs),
      HashBytes(state.trace_canonical_ids.data(),
                state.trace_canonical_ids.size() * sizeof(std::uint64_t)));
  const StateHash mesh_hash = CollectiveHash(local_mesh, state.ranks);
  const StateHash partition_hash =
      CollectiveHash(local_partition, state.ranks);
  const StateHash topology_hash = CollectiveHash(local_topology, state.ranks);
  PersistentResourceDescriptor resources;
  resources.mfem_pin = StableIdentity::Copy(expected_mfem_pin);
  resources.mesh_hash = mesh_hash;
  resources.partition_hash = partition_hash;
  resources.topology_hash = topology_hash;
  resources.rank_count = static_cast<std::uint32_t>(state.ranks);
  resources.polynomial_order = cpu_polynomial_order;
  resources.layout = StableIdentity::Copy(expected_layout);
  const std::size_t state_bytes =
      (state.Committed()->interior.capacity() +
       state.Committed()->trace.capacity()) * sizeof(double);
  resources.reuse = {true, true, true, false, 1, 1,
                     2 * state_bytes, state.WorkspaceBytes()};
  if (!resources.Valid()) { return false; }

  state.evidence.tuple = tuple;
  state.evidence.experiment = config.experiment;
  state.evidence.source = config.source;
  state.evidence.rank = state.rank;
  state.evidence.ranks = state.ranks;
  state.evidence.local_elements = state.mesh->GetNE();
  state.evidence.global_elements = static_cast<int>(SumU64(
      static_cast<std::uint64_t>(state.mesh->GetNE())));
  state.evidence.local_owned_interior_coefficients =
      state.Committed()->interior.size();
  state.evidence.global_owned_interior_coefficients = SumU64(
      state.evidence.local_owned_interior_coefficients);
  state.evidence.local_owned_true_trace_coefficients =
      state.Committed()->trace.size();
  state.evidence.global_owned_true_trace_coefficients = SumU64(
      state.evidence.local_owned_true_trace_coefficients);
  state.evidence.unique_periodic_faces =
      2U * cpu_elements_per_direction * cpu_elements_per_direction;
  state.evidence.negative_signed_trace_dofs = SumU64(
      static_cast<std::uint64_t>(std::count_if(
          state.signed_trace_dofs.begin(), state.signed_trace_dofs.end(),
          [](const int dof) { return dof < 0; })));
  state.evidence.mesh_hash = mesh_hash;
  state.evidence.partition_hash = partition_hash;
  state.evidence.topology_hash = topology_hash;
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
  state.evidence.storage.topology_generation = 1;
  state.evidence.storage.storage_generation = 1;
  state.evidence.storage.allocation_baseline = 1;
  state.evidence.storage.allocation_events_after_baseline = 0;

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
  CpuAttemptEvidence& evidence = state.active_evidence;
  evidence = {};
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
                  const CpuAttemptStatus status = CpuAttemptStatus::runtime_failure)
  {
    evidence.failure_phase = phase;
    evidence.status = status;
    evidence.finite_gate = false;
    evidence.admissibility_gate = false;
    evidence.recovery_gate = false;
    evidence.nonlinear_gate = false;
    evidence.true_residual_gate = false;
    result.evidence = evidence;
    result.gates = {false, false, false, false, true, false};
    result.originating_failure = FailureReason::backend_failure;
    return result;
  };

  if (!state.initialized ||
      !state.InjectionArguments(injection, coordinates.step + 1) ||
      !state.Committed() || !state.Candidate() ||
      state.Candidate()->interior != state.Committed()->interior ||
      state.Candidate()->trace != state.Committed()->trace)
  {
    return fail(CpuFailurePhase::setup, CpuAttemptStatus::invalid_request);
  }

  bool local_ok = AssignGridFunction(*state.old_interior,
                                     state.Committed()->interior);
  if (coordinates.step == 0)
  {
    local_ok = local_ok && AssignGridFunction(
        *state.iterate_interior, state.first_iterate_interior) &&
        Assign(state.iterate_true_trace, state.first_iterate_trace);
  }
  else
  {
    local_ok = local_ok && AssignGridFunction(
        *state.iterate_interior, state.Committed()->interior) &&
        Assign(state.iterate_true_trace, state.Committed()->trace);
  }
  local_ok = local_ok &&
      state.LoadTrace(state.iterate_true_trace, *state.iterate_trace);
  if (!state.CollectiveAll(local_ok))
  {
    return fail(CpuFailurePhase::setup);
  }

  double initial_density = 0.0;
  double initial_pressure = 0.0;
  double initial_wave = 0.0;
  if (!state.Prescreen(*state.iterate_interior, *state.iterate_trace,
                       initial_density, initial_pressure, initial_wave))
  {
    return fail(CpuFailurePhase::assembly);
  }
  if (!state.EnterPhase(injection, CpuFailurePhase::assembly))
  {
    return fail(CpuFailurePhase::assembly,
                CpuAttemptStatus::injected_failure);
  }
  if (!state.Assemble(*state.iterate_interior, *state.iterate_trace,
                      state.current_system))
  {
    return fail(CpuFailurePhase::assembly);
  }
  ++evidence.work.complete_assemblies;
  state.CompletePhase();
  double residual = state.ResidualNorm(state.current_system);
  if (!Finite(residual) || !(residual > 0.0))
  {
    return fail(CpuFailurePhase::assembly);
  }
  evidence.initial_nonlinear_residual = residual;
  evidence.final_nonlinear_residual = residual;
  evidence.products[0] = {DerivedProductKind::residual, attempt.sequence,
                          attempt.sequence,
                          HashPair(HashDoubles(state.current_system.residual_interior),
                                   HashDoubles(state.current_system.residual_trace)),
                          true};

  int updates = 0;
  bool recovered = false;
  double maximum_true_residual = 0.0;
  while (updates < state.config.maximum_newton_updates)
  {
    if (updates > 0 && residual <= state.config.nonlinear_tolerance *
                                      (1.0 + evidence.initial_nonlinear_residual))
    {
      break;
    }
    if (!state.EnterPhase(injection, CpuFailurePhase::condensation))
    {
      return fail(CpuFailurePhase::condensation,
                  CpuAttemptStatus::injected_failure);
    }
    StateHash matrix_hash{};
    if (!state.CondenseAndAssemble(state.current_system, matrix_hash))
    {
      return fail(CpuFailurePhase::condensation);
    }
    ++evidence.work.complete_condensations;
    state.CompletePhase();
    if (!evidence.products[1].produced)
    {
      evidence.products[1] = {DerivedProductKind::matrix, attempt.sequence,
                              attempt.sequence, matrix_hash, true};
    }

    if (!state.EnterPhase(injection, CpuFailurePhase::trace_solve))
    {
      return fail(CpuFailurePhase::trace_solve,
                  CpuAttemptStatus::injected_failure);
    }
    double true_residual = 0.0;
    int linear_iterations = 0;
    StateHash factor_hash{}, solution_hash{};
    if (!state.SolveTrace(true_residual, linear_iterations, factor_hash,
                          solution_hash))
    {
      return fail(CpuFailurePhase::trace_solve);
    }
    evidence.linear_iterations += linear_iterations;
    maximum_true_residual = std::max(maximum_true_residual, true_residual);
    ++evidence.work.complete_trace_solves;
    state.CompletePhase();
    if (!evidence.products[2].produced)
    {
      evidence.products[2] = {DerivedProductKind::factor, attempt.sequence,
                              attempt.sequence, factor_hash, true};
      evidence.products[3] = {DerivedProductKind::solution, attempt.sequence,
                              attempt.sequence, solution_hash, true};
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

    bool accepted = false;
    for (int trial = 0;
         trial < state.config.maximum_line_search_trials; ++trial)
    {
      const double alpha = std::ldexp(1.0, -trial);
      CpuTrialEvidence& trial_evidence = evidence.trials[evidence.trial_count++];
      trial_evidence.newton_iteration = updates;
      trial_evidence.trial_index = trial;
      trial_evidence.alpha = alpha;
      trial_evidence.base_residual = residual;
      trial_evidence.armijo_boundary =
          (1.0 - state.config.armijo_constant * alpha) * residual;
      if (!state.MakeTrial(alpha)) { return fail(CpuFailurePhase::recovery); }
      double density = 0.0;
      double pressure = 0.0;
      double wave = 0.0;
      trial_evidence.prescreen_passed = state.Prescreen(
          *state.trial_interior, *state.trial_trace, density, pressure, wave);
      trial_evidence.minimum_density = density;
      trial_evidence.minimum_pressure = pressure;
      trial_evidence.maximum_wave_speed = wave;
      if (!trial_evidence.prescreen_passed) { continue; }
      if (!state.Assemble(*state.trial_interior, *state.trial_trace,
                          state.trial_system))
      {
        return fail(CpuFailurePhase::assembly);
      }
      trial_evidence.reassembled = true;
      ++evidence.work.complete_assemblies;
      ++evidence.work.trial_reassemblies;
      const double trial_residual = state.ResidualNorm(state.trial_system);
      trial_evidence.trial_residual = trial_residual;
      if (!Finite(trial_residual)) { return fail(CpuFailurePhase::assembly); }
      if (trial_residual <= trial_evidence.armijo_boundary)
      {
        trial_evidence.accepted = true;
        if (!state.AcceptTrial()) { return fail(CpuFailurePhase::recovery); }
        residual = trial_residual;
        evidence.minimum_density = density;
        evidence.minimum_pressure = pressure;
        evidence.maximum_wave_speed = wave;
        accepted = true;
        break;
      }
    }
    if (!accepted) { return fail(CpuFailurePhase::recovery); }
    ++updates;
    ++evidence.work.accepted_newton_updates;
  }

  evidence.nonlinear_updates = updates;
  evidence.final_nonlinear_residual = residual;
  evidence.maximum_true_relative_residual = maximum_true_residual;
  const bool nonlinear_converged = updates > 0 &&
      residual <= state.config.nonlinear_tolerance *
          (1.0 + evidence.initial_nonlinear_residual);
  if (!nonlinear_converged || !recovered)
  {
    return fail(CpuFailurePhase::recovery);
  }
  if (!state.StoreCandidate()) { return fail(CpuFailurePhase::recovery); }

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
  if (!state.Prescreen(*state.iterate_interior, *state.iterate_trace,
                       final_density, final_pressure, final_wave))
  {
    return fail(CpuFailurePhase::diagnostics);
  }
  state.CompletePhase();

  evidence.minimum_density = final_density;
  evidence.minimum_pressure = final_pressure;
  evidence.maximum_wave_speed = final_wave;
  evidence.finite_gate = Finite(residual) && Finite(maximum_true_residual) &&
      Finite(state.Candidate()->interior) && Finite(state.Candidate()->trace);
  evidence.admissibility_gate = final_density >= density_gate &&
      final_pressure >= pressure_gate && final_wave < cpu_fixed_tau;
  evidence.recovery_gate = recovered;
  evidence.nonlinear_gate = nonlinear_converged;
  evidence.true_residual_gate = maximum_true_residual <=
      state.config.true_residual_tolerance;
  evidence.status = CpuAttemptStatus::success;
  evidence.failure_phase = CpuFailurePhase::none;
  result.evidence = evidence;
  result.gates = {evidence.finite_gate, evidence.admissibility_gate,
                  evidence.recovery_gate, evidence.true_residual_gate,
                  true, evidence.nonlinear_gate};
  result.originating_failure = FailureReason::none;
  return result;
}

void detail::CpuNumericsCore::Complete(
    const AttemptOutcome& outcome,
    const StateCoordinates& coordinates,
    const RunTelemetry&) noexcept
{
  Implementation& state = *implementation_;
  CpuAttemptEvidence& attempt = state.active_evidence;
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
    state.evidence.accepted = {true, coordinates,
                               attempt.committed_interior_after,
                               attempt.committed_trace_after, attempt};
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
  state.evidence.storage.allocation_events_after_baseline = 0;
}

bool detail::CpuNumericsCore::CaptureCheckpoint(
    const CheckpointDescriptor& descriptor,
    CpuCheckpoint& checkpoint) noexcept
{
  Implementation& state = *implementation_;
  if (!state.initialized || !descriptor.successful_commit) { return false; }
  if (!ValidSlot(checkpoint.state))
  {
    checkpoint.state = state.checkpoint_slot;
  }
  if (!CopySlot(state.committed_slot, checkpoint.state)) { return false; }
  checkpoint.descriptor = descriptor;
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
  state.evidence.accepted = {true, coordinates,
                             restored.committed_interior_after,
                             restored.committed_trace_after, restored};
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
  snapshot.interior = {storage->interior.data(), storage->interior.size(),
                       storage->interior.capacity()};
  snapshot.trace = {storage->trace.data(), storage->trace.size(),
                    storage->trace.capacity()};
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
  snapshot.interior = {storage->interior.data(), storage->interior.size(),
                       storage->interior.capacity()};
  snapshot.trace = {storage->trace.data(), storage->trace.size(),
                    storage->trace.capacity()};
  return snapshot;
}

void detail::CpuNumericsCore::MarkPreparedFailure(
    const CpuFailurePhase phase, const int source_rank) noexcept
{
  implementation_->active_evidence.failure_phase = phase;
  implementation_->active_evidence.failure_source_rank = source_rank;
  implementation_->active_evidence.status = CpuAttemptStatus::runtime_failure;
}

const CpuRunEvidence& detail::CpuNumericsCore::evidence() const noexcept
{
  return implementation_->evidence;
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

} // namespace dg::hdg::runtime::cpu

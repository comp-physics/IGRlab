#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{
using dg::hdg::runtime::cuda_w8a::detail::CudaLaunchHash;
using dg::hdg::runtime::cuda_w8a::detail::CudaLaunchHashPair;

constexpr std::array<std::uint64_t, 4> hash_seeds{{
    1469598103934665603ULL,
    1099511628211ULL ^ 0x9e3779b97f4a7c15ULL,
    1469598103934665603ULL ^ 0xd6e8feb86659fd93ULL,
    1099511628211ULL ^ 0xa0761d6478bd642fULL}};

std::array<std::uint64_t, 4> HistoricalByteHash(
    const double* values, const int count) noexcept
{
  std::array<std::uint64_t, 4> hashes = hash_seeds;
  const auto* bytes = reinterpret_cast<const unsigned char*>(values);
  const int byte_count = count * static_cast<int>(sizeof(double));
  for (int lane = 0; lane < 4; ++lane)
    for (int index = 0; index < byte_count; ++index)
    {
      hashes[lane] ^= static_cast<std::uint64_t>(
          bytes[index] + 17U * static_cast<unsigned int>(lane));
      hashes[lane] *= 1099511628211ULL +
          2ULL * static_cast<std::uint64_t>(lane);
    }
  return hashes;
}

template <typename T>
class DeviceAllocation
{
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation()
  {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
  }

  bool Allocate(const std::size_t count) noexcept
  {
    return pointer_ == nullptr && count > 0 &&
        cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)) ==
            cudaSuccess;
  }
  T* data() noexcept { return pointer_; }
  const T* data() const noexcept { return pointer_; }

 private:
  T* pointer_ = nullptr;
};

bool CopyHashes(std::uint64_t* device_words, const std::size_t count,
                std::array<std::uint64_t, 8>& words) noexcept
{
  words.fill(0);
  return cudaDeviceSynchronize() == cudaSuccess &&
      cudaMemcpy(words.data(), device_words, count * sizeof(std::uint64_t),
                 cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool CompareHash(const std::array<std::uint64_t, 4>& expected,
                 const std::uint64_t* actual, const char* launch,
                 const int first_count, const int second_count = -1) noexcept
{
  for (int lane = 0; lane < 4; ++lane)
    if (expected[static_cast<std::size_t>(lane)] != actual[lane])
    {
      std::cerr << "CUDA exact hash mismatch launch=" << launch
                << " first_count=" << first_count;
      if (second_count >= 0) std::cerr << " second_count=" << second_count;
      std::cerr << " lane=" << lane << " expected=0x" << std::hex
                << expected[static_cast<std::size_t>(lane)] << " actual=0x"
                << actual[lane] << std::dec << '\n';
      return false;
    }
  return true;
}

bool CheckSingle(const double* host_values, const double* device_values,
                 const int count, std::uint64_t* device_words) noexcept
{
  if (!CudaLaunchHash(device_values, count, device_words)) return false;
  std::array<std::uint64_t, 8> actual{};
  return CopyHashes(device_words, 4, actual) &&
      CompareHash(HistoricalByteHash(host_values, count), actual.data(),
                  "single", count);
}

bool CheckPair(const double* first_host, const int first_count,
               const double* first_device, const double* second_host,
               const int second_count, const double* second_device,
               std::uint64_t* device_words) noexcept
{
  if (!CudaLaunchHashPair(first_device, first_count, second_device,
                          second_count, device_words))
    return false;
  std::array<std::uint64_t, 8> actual{};
  if (!CopyHashes(device_words, 8, actual)) return false;
  return CompareHash(HistoricalByteHash(first_host, first_count),
                     actual.data(), "pair-first", first_count, second_count) &&
      CompareHash(HistoricalByteHash(second_host, second_count),
                  actual.data() + 4, "pair-second", first_count, second_count);
}

std::vector<double> MakeExactValues()
{
  std::vector<std::uint64_t> bits{{
      0x0000000000000000ULL, // +0
      0x8000000000000000ULL, // -0
      0x3ff0000000000000ULL,
      0xbff0000000000000ULL,
      0x7ff0000000000000ULL, // +infinity
      0xfff0000000000000ULL, // -infinity
      0x7ff8000000000001ULL, // quiet NaN payloads and signs
      0x7ff8deadbeef1234ULL,
      0xfff8abcdef012345ULL,
      0x7ff0000000000001ULL, // signaling NaN payload
      0x0000000000000001ULL,
      0x8000000000000001ULL,
      0x7fefffffffffffffULL,
      0xffefffffffffffffULL,
      0x0123456789abcdefULL,
      0xfedcba9876543210ULL}};
  std::uint64_t state = 0x6a09e667f3bcc909ULL;
  while (bits.size() < 193)
  {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    bits.push_back(state);
  }
  std::vector<double> values(bits.size());
  std::memcpy(values.data(), bits.data(), bits.size() * sizeof(std::uint64_t));
  return values;
}

} // namespace

int main()
{
  const std::uint64_t endian_probe = 1;
  if (*reinterpret_cast<const unsigned char*>(&endian_probe) != 1)
  {
    std::cerr << "CUDA exact hash fixture requires little-endian host bytes\n";
    return 2;
  }

  const std::vector<double> values = MakeExactValues();
  DeviceAllocation<double> device_values;
  DeviceAllocation<std::uint64_t> device_words;
  if (!device_values.Allocate(values.size()) || !device_words.Allocate(8) ||
      cudaMemcpy(device_values.data(), values.data(),
                 values.size() * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess)
    return 3;

  constexpr std::array<int, 22> counts{{
      0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
      127, 128, 129, 193}};
  constexpr std::array<int, 3> offsets{{0, 1, 3}};
  for (const int offset : offsets)
    for (const int count : counts)
      if (static_cast<std::size_t>(offset + count) <= values.size() &&
          !CheckSingle(values.data() + offset,
                       device_values.data() + offset, count,
                       device_words.data()))
        return 4;
  if (!CheckSingle(nullptr, nullptr, 0, device_words.data())) return 5;

  struct PairCase
  {
    int first_offset;
    int first_count;
    int second_offset;
    int second_count;
  };
  constexpr std::array<PairCase, 10> pairs{{
      {0, 0, 0, 0}, {0, 0, 1, 17}, {3, 17, 0, 0},
      {0, 1, 1, 3}, {1, 5, 3, 7}, {3, 17, 1, 33},
      {0, 32, 1, 31}, {1, 65, 3, 64}, {3, 127, 1, 129},
      {1, 189, 0, 193}}};
  for (const PairCase& pair : pairs)
  {
    const bool first_empty = pair.first_count == 0;
    const bool second_empty = pair.second_count == 0;
    const double* first_host = first_empty
        ? nullptr : values.data() + pair.first_offset;
    const double* first_device = first_empty
        ? nullptr : device_values.data() + pair.first_offset;
    const double* second_host = second_empty
        ? nullptr : values.data() + pair.second_offset;
    const double* second_device = second_empty
        ? nullptr : device_values.data() + pair.second_offset;
    if (!CheckPair(first_host, pair.first_count, first_device,
                   second_host, pair.second_count, second_device,
                   device_words.data()))
      return 6;
  }
  return 0;
}

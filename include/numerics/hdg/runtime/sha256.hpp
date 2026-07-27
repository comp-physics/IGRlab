#pragma once

/**
 * @file numerics/hdg/runtime/sha256.hpp
 * @brief Dependency-free C++17 FIPS 180-4 SHA-256 interface.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace dg::hdg::runtime
{

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256 final
{
public:
  Sha256() noexcept;

  /** Update the digest. A nonzero length requires a non-null input. */
  bool Update(const void* data, std::size_t length) noexcept;
  bool Update(std::string_view text) noexcept
  {
    return Update(text.data(), text.size());
  }

  /** Finalize once. Repeated calls return the same digest. */
  bool Finalize(Sha256Digest& digest) noexcept;

private:
  void Transform(const std::uint8_t block[64]) noexcept;

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> buffer_{};
  Sha256Digest digest_{};
  std::uint64_t total_bytes_ = 0;
  std::size_t buffered_bytes_ = 0;
  bool finalized_ = false;
  bool valid_ = true;
};

/** One-shot SHA-256. Invalid pointer/length input returns the zero digest. */
Sha256Digest ComputeSha256(const void* data, std::size_t length) noexcept;
Sha256Digest ComputeSha256(std::string_view text) noexcept;

} // namespace dg::hdg::runtime

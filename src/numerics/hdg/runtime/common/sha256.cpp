#include "numerics/hdg/runtime/sha256.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace dg::hdg::runtime
{
namespace
{

constexpr std::array<std::uint32_t, 64> round_constants{{
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

constexpr std::uint32_t RotateRight(const std::uint32_t value,
                                    const unsigned int amount) noexcept
{
  return (value >> amount) | (value << (32U - amount));
}

constexpr std::uint32_t Choose(const std::uint32_t x,
                               const std::uint32_t y,
                               const std::uint32_t z) noexcept
{
  return (x & y) ^ (~x & z);
}

constexpr std::uint32_t Majority(const std::uint32_t x,
                                 const std::uint32_t y,
                                 const std::uint32_t z) noexcept
{
  return (x & y) ^ (x & z) ^ (y & z);
}

constexpr std::uint32_t BigSigma0(const std::uint32_t value) noexcept
{
  return RotateRight(value, 2U) ^ RotateRight(value, 13U) ^
      RotateRight(value, 22U);
}

constexpr std::uint32_t BigSigma1(const std::uint32_t value) noexcept
{
  return RotateRight(value, 6U) ^ RotateRight(value, 11U) ^
      RotateRight(value, 25U);
}

constexpr std::uint32_t SmallSigma0(const std::uint32_t value) noexcept
{
  return RotateRight(value, 7U) ^ RotateRight(value, 18U) ^ (value >> 3U);
}

constexpr std::uint32_t SmallSigma1(const std::uint32_t value) noexcept
{
  return RotateRight(value, 17U) ^ RotateRight(value, 19U) ^ (value >> 10U);
}

std::uint32_t LoadBigEndian32(const std::uint8_t* bytes) noexcept
{
  return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
      (static_cast<std::uint32_t>(bytes[1]) << 16U) |
      (static_cast<std::uint32_t>(bytes[2]) << 8U) |
      static_cast<std::uint32_t>(bytes[3]);
}

void StoreBigEndian32(const std::uint32_t value,
                      std::uint8_t* bytes) noexcept
{
  bytes[0] = static_cast<std::uint8_t>(value >> 24U);
  bytes[1] = static_cast<std::uint8_t>(value >> 16U);
  bytes[2] = static_cast<std::uint8_t>(value >> 8U);
  bytes[3] = static_cast<std::uint8_t>(value);
}

void StoreBigEndian64(const std::uint64_t value,
                      std::uint8_t* bytes) noexcept
{
  for (std::size_t index = 0; index < 8; ++index)
  {
    bytes[index] = static_cast<std::uint8_t>(value >> (56U - 8U * index));
  }
}

} // namespace

Sha256::Sha256() noexcept
    : state_{{0x6a09e667U,
              0xbb67ae85U,
              0x3c6ef372U,
              0xa54ff53aU,
              0x510e527fU,
              0x9b05688cU,
              0x1f83d9abU,
              0x5be0cd19U}}
{}

bool Sha256::Update(const void* data, const std::size_t length) noexcept
{
  if (!valid_ || finalized_ || (length != 0 && data == nullptr))
  {
    valid_ = false;
    return false;
  }
  constexpr std::uint64_t maximum_bytes =
      std::numeric_limits<std::uint64_t>::max() / 8U;
  if (length > maximum_bytes ||
      static_cast<std::uint64_t>(length) > maximum_bytes - total_bytes_)
  {
    valid_ = false;
    return false;
  }

  const auto* input = static_cast<const std::uint8_t*>(data);
  total_bytes_ += static_cast<std::uint64_t>(length);
  std::size_t consumed = 0;

  if (buffered_bytes_ != 0)
  {
    const std::size_t copied =
        std::min(length, buffer_.size() - buffered_bytes_);
    if (copied != 0)
    {
      std::memcpy(buffer_.data() + buffered_bytes_, input, copied);
      buffered_bytes_ += copied;
      consumed += copied;
    }
    if (buffered_bytes_ == buffer_.size())
    {
      Transform(buffer_.data());
      buffered_bytes_ = 0;
    }
  }

  while (length - consumed >= buffer_.size())
  {
    Transform(input + consumed);
    consumed += buffer_.size();
  }

  const std::size_t remaining = length - consumed;
  if (remaining != 0)
  {
    std::memcpy(buffer_.data(), input + consumed, remaining);
    buffered_bytes_ = remaining;
  }
  return true;
}

bool Sha256::Finalize(Sha256Digest& digest) noexcept
{
  if (!valid_) { return false; }
  if (finalized_)
  {
    digest = digest_;
    return true;
  }

  buffer_[buffered_bytes_++] = 0x80U;
  if (buffered_bytes_ > 56)
  {
    std::fill(buffer_.begin() + buffered_bytes_, buffer_.end(), 0U);
    Transform(buffer_.data());
    buffered_bytes_ = 0;
  }
  std::fill(buffer_.begin() + buffered_bytes_, buffer_.begin() + 56, 0U);
  StoreBigEndian64(total_bytes_ * 8U, buffer_.data() + 56);
  Transform(buffer_.data());

  for (std::size_t index = 0; index < state_.size(); ++index)
  {
    StoreBigEndian32(state_[index], digest_.data() + 4 * index);
  }
  finalized_ = true;
  buffered_bytes_ = 0;
  digest = digest_;
  return true;
}

void Sha256::Transform(const std::uint8_t block[64]) noexcept
{
  std::array<std::uint32_t, 64> words{};
  for (std::size_t index = 0; index < 16; ++index)
  {
    words[index] = LoadBigEndian32(block + 4 * index);
  }
  for (std::size_t index = 16; index < words.size(); ++index)
  {
    words[index] = SmallSigma1(words[index - 2]) + words[index - 7] +
        SmallSigma0(words[index - 15]) + words[index - 16];
  }

  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  std::uint32_t e = state_[4];
  std::uint32_t f = state_[5];
  std::uint32_t g = state_[6];
  std::uint32_t h = state_[7];

  for (std::size_t index = 0; index < words.size(); ++index)
  {
    const std::uint32_t first = h + BigSigma1(e) + Choose(e, f, g) +
        round_constants[index] + words[index];
    const std::uint32_t second = BigSigma0(a) + Majority(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + first;
    d = c;
    c = b;
    b = a;
    a = first + second;
  }

  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

Sha256Digest ComputeSha256(const void* data,
                           const std::size_t length) noexcept
{
  Sha256 hash;
  Sha256Digest digest{};
  if (!hash.Update(data, length) || !hash.Finalize(digest)) { return {}; }
  return digest;
}

Sha256Digest ComputeSha256(const std::string_view text) noexcept
{
  return ComputeSha256(text.data(), text.size());
}

} // namespace dg::hdg::runtime

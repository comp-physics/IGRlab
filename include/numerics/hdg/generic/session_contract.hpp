#pragma once

/**
 * @file numerics/hdg/generic/session_contract.hpp
 * @brief Backend-neutral value records for a staged generic Euler HDG session.
 *
 * This is a numerical boundary only.  It owns no MFEM object, numerical
 * operator, output writer, restart payload, or virtual dispatch surface.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace dg::hdg::generic
{

/**
 * Caller-owned host views of the semantic Euler HDG interior fields.
 *
 * The conservative coefficients contain four component-major, byNODES L2
 * slabs.  A non-null @ref sigma_coefficients requests the scalar IGR field;
 * the private auxiliary gradient and every trace field remain unexposed.
 * Backends may use any internal layout, but must fill both requested views
 * from the same committed or prepared state transaction.
 */
struct HdgEulerHostInteriorFields
{
  double* conservative_coefficients = nullptr;
  double* sigma_coefficients = nullptr;
  std::size_t scalar_count = 0;
};

enum class HdgEulerSessionCapability : std::uint32_t
{
  prepared_commit = 1u << 0,
  prepared_output = 1u << 1,
  committed_cfl_diagnostics = 1u << 2,
  generic_borrowed_inputs = 1u << 3
};

constexpr std::uint32_t HdgEulerSessionCapabilityBit(
    const HdgEulerSessionCapability capability) noexcept
{
  return static_cast<std::uint32_t>(capability);
}

constexpr bool HasHdgEulerSessionCapabilities(
    const std::uint32_t available, const std::uint32_t required) noexcept
{
  return (available & required) == required;
}

struct HdgEulerSessionCoordinates
{
  double time = 0.0;
  double dt = 0.0;
  std::uint64_t step = 0;
  std::uint64_t generation = 0;
};

/**
 * Committed diagnostics only.  An unavailable CFL estimate is represented by
 * a NaN @ref backend_stable_dt and no committed_cfl_diagnostics bit.
 */
struct HdgEulerCommittedDiagnostics
{
  bool available = false;
  HdgEulerSessionCoordinates coordinates{};
  std::uint64_t wave_generation = 0;
  double maximum_wave_speed = 0.0;
  double backend_stable_dt = std::numeric_limits<double>::quiet_NaN();
};

enum class HdgEulerSessionStatus : std::uint8_t
{
  uninitialized,
  initialized,
  prepared,
  committed,
  aborted,
  rejected
};

struct HdgEulerSessionResult
{
  HdgEulerSessionStatus status = HdgEulerSessionStatus::uninitialized;
  HdgEulerSessionCoordinates coordinates{};
};

inline bool IsValidHdgEulerSessionCoordinates(
    const HdgEulerSessionCoordinates coordinates) noexcept
{
  return std::isfinite(coordinates.time) && std::isfinite(coordinates.dt) &&
      coordinates.time >= 0.0 && coordinates.dt >= 0.0;
}

inline bool IsValidHdgEulerCommittedDiagnostics(
    const HdgEulerCommittedDiagnostics diagnostics,
    const std::uint32_t capabilities) noexcept
{
  if (!diagnostics.available ||
      !IsValidHdgEulerSessionCoordinates(diagnostics.coordinates) ||
      diagnostics.wave_generation != diagnostics.coordinates.generation)
  {
    return false;
  }
  const bool cfl_available = HasHdgEulerSessionCapabilities(
      capabilities,
      HdgEulerSessionCapabilityBit(
          HdgEulerSessionCapability::committed_cfl_diagnostics));
  const bool wave_available = std::isfinite(diagnostics.maximum_wave_speed) &&
      diagnostics.maximum_wave_speed >= 0.0;
  if (cfl_available)
  {
    return wave_available && std::isfinite(diagnostics.backend_stable_dt) &&
        diagnostics.backend_stable_dt > 0.0;
  }
  // Generation zero may not have a measured wave value.  NaN is the sole
  // unavailable spelling; infinities remain invalid diagnostics.
  const bool wave_unavailable_at_generation_zero =
      diagnostics.coordinates.generation == 0 &&
      std::isnan(diagnostics.maximum_wave_speed);
  return (wave_available || wave_unavailable_at_generation_zero) &&
      std::isnan(diagnostics.backend_stable_dt);
}

static_assert(std::is_standard_layout<HdgEulerSessionCoordinates>::value &&
                  std::is_trivially_copyable<HdgEulerSessionCoordinates>::value,
              "generic HDG session coordinates must remain POD");
static_assert(std::is_standard_layout<HdgEulerCommittedDiagnostics>::value &&
                  std::is_trivially_copyable<HdgEulerCommittedDiagnostics>::value,
              "generic HDG committed diagnostics must remain POD");
static_assert(std::is_standard_layout<HdgEulerSessionResult>::value &&
                  std::is_trivially_copyable<HdgEulerSessionResult>::value,
              "generic HDG session result must remain POD");

} // namespace dg::hdg::generic

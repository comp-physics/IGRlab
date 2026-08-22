#pragma once

/**
 * @file numerics/hdg/generic/session_contract.hpp
 * @brief Backend-neutral value records for a staged generic Euler HDG session.
 *
 * This is a numerical boundary only.  It owns no MFEM object, numerical
 * operator, output writer, restart payload, or virtual dispatch surface.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

namespace dg::hdg::generic
{

enum class HdgEulerSessionCapability : std::uint32_t
{
  prepared_commit = 1u << 0,
  prepared_output = 1u << 1,
  committed_cfl_diagnostics = 1u << 2,
  generic_borrowed_inputs = 1u << 3,
  committed_observables = 1u << 4,
  committed_nonlinear_residual_audit = 1u << 5
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

/**
 * Caller-triggered collective summary of the committed volume state.  This
 * is a single fresh snapshot, never a cached or per-step history record.
 */
struct HdgEulerCommittedObservables
{
  bool available = false;
  HdgEulerSessionCoordinates coordinates{};
  int quadrature_order = 0;
  std::array<double, 4> conserved_integrals{};
  std::array<double, 4> conserved_l2_norms{};
  double minimum_density = 0.0;
  double minimum_pressure = 0.0;
};

/** The residual vectors differ between the two admitted Euler formulations. */
enum class HdgEulerResidualFormulation : std::uint8_t
{
  unavailable = 0,
  conventional_u4_uhat4 = 1,
  traced_scalar_igr_p1_u7_uhat5 = 2
};

/** One generation-bound committed nonlinear-residual audit. */
struct HdgEulerCommittedResidualAudit
{
  bool available = false;
  HdgEulerSessionCoordinates coordinates{};
  std::uint64_t residual_generation = 0;
  int quadrature_order = 0;
  HdgEulerResidualFormulation formulation =
      HdgEulerResidualFormulation::unavailable;
  double norm = std::numeric_limits<double>::quiet_NaN();
};

constexpr bool IsKnownHdgEulerResidualFormulation(
    const HdgEulerResidualFormulation formulation) noexcept
{
  return formulation == HdgEulerResidualFormulation::conventional_u4_uhat4 ||
      formulation ==
          HdgEulerResidualFormulation::traced_scalar_igr_p1_u7_uhat5;
}

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

inline bool IsValidHdgEulerCommittedObservables(
    const HdgEulerCommittedObservables observables,
    const std::uint32_t capabilities) noexcept
{
  return observables.available &&
      HasHdgEulerSessionCapabilities(
          capabilities, HdgEulerSessionCapabilityBit(
                            HdgEulerSessionCapability::committed_observables)) &&
      IsValidHdgEulerSessionCoordinates(observables.coordinates) &&
      observables.quadrature_order > 0 &&
      std::all_of(observables.conserved_integrals.begin(),
                  observables.conserved_integrals.end(),
                  [](const double value) { return std::isfinite(value); }) &&
      std::all_of(observables.conserved_l2_norms.begin(),
                  observables.conserved_l2_norms.end(),
                  [](const double value)
                  { return std::isfinite(value) && value >= 0.0; }) &&
      std::isfinite(observables.minimum_density) &&
      std::isfinite(observables.minimum_pressure);
}

inline bool IsValidHdgEulerCommittedResidualAudit(
    const HdgEulerCommittedResidualAudit audit,
    const std::uint32_t capabilities) noexcept
{
  if (!HasHdgEulerSessionCapabilities(
          capabilities, HdgEulerSessionCapabilityBit(
                            HdgEulerSessionCapability::committed_nonlinear_residual_audit)) ||
      !IsValidHdgEulerSessionCoordinates(audit.coordinates) ||
      audit.residual_generation != audit.coordinates.generation ||
      audit.quadrature_order <= 0)
  {
    return false;
  }
  return audit.available && IsKnownHdgEulerResidualFormulation(audit.formulation) &&
      std::isfinite(audit.norm) && audit.norm >= 0.0;
}

/** Internal lifecycle coherence also admits explicit unavailable evidence. */
inline bool IsCoherentHdgEulerCommittedResidualAudit(
    const HdgEulerCommittedResidualAudit audit) noexcept
{
  return IsValidHdgEulerSessionCoordinates(audit.coordinates) &&
      audit.residual_generation == audit.coordinates.generation &&
      audit.quadrature_order > 0 &&
      (audit.available
           ? IsKnownHdgEulerResidualFormulation(audit.formulation) &&
                 std::isfinite(audit.norm) && audit.norm >= 0.0
           : audit.formulation == HdgEulerResidualFormulation::unavailable &&
                 std::isnan(audit.norm));
}

static_assert(std::is_standard_layout<HdgEulerSessionCoordinates>::value &&
                  std::is_trivially_copyable<HdgEulerSessionCoordinates>::value,
              "generic HDG session coordinates must remain POD");
static_assert(std::is_standard_layout<HdgEulerCommittedDiagnostics>::value &&
                  std::is_trivially_copyable<HdgEulerCommittedDiagnostics>::value,
              "generic HDG committed diagnostics must remain POD");
static_assert(std::is_standard_layout<HdgEulerCommittedObservables>::value &&
                  std::is_trivially_copyable<HdgEulerCommittedObservables>::value,
              "generic HDG committed observables must remain POD");
static_assert(std::is_standard_layout<HdgEulerCommittedResidualAudit>::value &&
                  std::is_trivially_copyable<HdgEulerCommittedResidualAudit>::value,
              "generic HDG committed residual audit must remain POD");
static_assert(std::is_standard_layout<HdgEulerSessionResult>::value &&
                  std::is_trivially_copyable<HdgEulerSessionResult>::value,
              "generic HDG session result must remain POD");

} // namespace dg::hdg::generic

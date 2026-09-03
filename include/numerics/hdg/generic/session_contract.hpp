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
 * Backends may use any internal layout, but must fill every requested view
 * from the same committed or prepared state transaction.  The optional
 * Zhang--Shu arrays are element ordered, not finite-element coefficient
 * arrays.  They retain the minimum density and pressure factors observed by
 * the bounded limiter calls across accepted RODAS attempts since session
 * initialization.  Rejected attempts never publish and no stage history is
 * exposed.  This running accepted-history aggregate prevents activations on
 * substeps between VTK output times from disappearing before publication.
 */
struct HdgEulerHostInteriorFields
{
  double* conservative_coefficients = nullptr;
  double* sigma_coefficients = nullptr;
  std::size_t scalar_count = 0;
  double* zhang_shu_theta_rho = nullptr;
  double* zhang_shu_theta_p = nullptr;
  std::size_t zhang_shu_element_count = 0;
};

enum class HdgEulerSessionCapability : std::uint32_t
{
  prepared_commit = 1u << 0,
  prepared_output = 1u << 1,
  committed_cfl_diagnostics = 1u << 2,
  generic_borrowed_inputs = 1u << 3,
  prepared_embedded_assessment = 1u << 4
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
 * Scale and budget used to assess one embedded prepared candidate.
 *
 * This is a control contract, not a diagnostic stream: it contains only the
 * three values needed for the collective accept/reject decision.  The same
 * absolute/relative scale is applied pointwise to differential U4 in the
 * residual volume quadrature and coefficientwise to the final algebraic
 * reconstruction.  The latter is required to consume no more than
 * @ref algebraic_correction_fraction of the temporal-error budget.
 */
struct HdgEulerPreparedAssessmentPolicy
{
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  double algebraic_correction_fraction = 0.0;
};

/**
 * Bounded aggregate used by adaptive time control for one live prepared step.
 * No stage, iteration, callback, or raw residual history is retained here.
 */
struct HdgEulerPreparedAssessment
{
  double normalized_differential_error =
      std::numeric_limits<double>::infinity();
  double normalized_algebraic_correction =
      std::numeric_limits<double>::infinity();
  bool finite_gate = false;
  bool solver_gate = false;
  bool residual_gate = false;
  bool admissibility_gate = false;
  bool algebraic_closure_gate = false;
  bool algebraic_correction_gate = false;
};

inline bool IsValidHdgEulerPreparedAssessmentPolicy(
    const HdgEulerPreparedAssessmentPolicy policy) noexcept
{
  return std::isfinite(policy.absolute_tolerance) &&
      policy.absolute_tolerance >= 0.0 &&
      std::isfinite(policy.relative_tolerance) &&
      policy.relative_tolerance >= 0.0 &&
      (policy.absolute_tolerance > 0.0 ||
       policy.relative_tolerance > 0.0) &&
      std::isfinite(policy.algebraic_correction_fraction) &&
      policy.algebraic_correction_fraction > 0.0 &&
      policy.algebraic_correction_fraction <= 1.0;
}

inline bool IsValidHdgEulerPreparedAssessment(
    const HdgEulerPreparedAssessment assessment) noexcept
{
  return std::isfinite(assessment.normalized_differential_error) &&
      assessment.normalized_differential_error >= 0.0 &&
      std::isfinite(assessment.normalized_algebraic_correction) &&
      assessment.normalized_algebraic_correction >= 0.0;
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

static_assert(std::is_standard_layout<HdgEulerSessionCoordinates>::value &&
                  std::is_trivially_copyable<HdgEulerSessionCoordinates>::value,
              "generic HDG session coordinates must remain POD");
static_assert(std::is_standard_layout<HdgEulerCommittedDiagnostics>::value &&
                  std::is_trivially_copyable<HdgEulerCommittedDiagnostics>::value,
              "generic HDG committed diagnostics must remain POD");
static_assert(
    std::is_standard_layout<HdgEulerPreparedAssessmentPolicy>::value &&
        std::is_trivially_copyable<
            HdgEulerPreparedAssessmentPolicy>::value,
    "generic HDG prepared assessment policy must remain POD");
static_assert(std::is_standard_layout<HdgEulerPreparedAssessment>::value &&
                  std::is_trivially_copyable<
                      HdgEulerPreparedAssessment>::value,
              "generic HDG prepared assessment must remain POD");
static_assert(std::is_standard_layout<HdgEulerSessionResult>::value &&
                  std::is_trivially_copyable<HdgEulerSessionResult>::value,
              "generic HDG session result must remain POD");

} // namespace dg::hdg::generic

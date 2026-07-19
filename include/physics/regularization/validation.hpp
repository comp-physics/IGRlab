#pragma once

#include "physics/regularization/auxiliary_field.hpp"
#include "physics/regularization/axes.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <type_traits>

namespace dg_igr
{
namespace regularization
{

enum class ValidationCode : std::uint8_t
{
  Valid,
  UnknownAxisValue,
  UnsupportedCombination,
  BackendUnavailable,
  InvalidDimension,
  InvalidComponentCount,
  InvalidPointCount,
  MissingBuffer,
  InvalidBufferSize,
  InvalidFieldIdentity,
  InvalidFieldTopology,
  MissingFieldCapability,
  StaleProvenance
};

struct ValidationResult
{
  ValidationCode code = ValidationCode::Valid;
  const char* message = "valid";

  constexpr bool Ok() const { return code == ValidationCode::Valid; }
};

struct RegularizationValidationContext
{
  bool mfem_device_available = false;
  int expected_rk_stage = -1;
  std::uint64_t expected_state_generation = 0u;
};

/** Typed binding for an equation-by-direction volume-flux correction. */
struct VolumeFluxCorrectionFieldBinding
{
  mfem::Vector* storage = nullptr;
  int components = 0;
  int dimension = 0;
  int point_count = 0;

  int ExpectedSize() const
  {
    return components * dimension * point_count;
  }
  bool HasExpectedSize() const
  {
    return storage != nullptr && storage->Size() == ExpectedSize();
  }
};

/** Typed binding for an equation-vector normal or wall correction. */
struct StateVectorCorrectionFieldBinding
{
  mfem::Vector* storage = nullptr;
  int components = 0;
  int point_count = 0;

  int ExpectedSize() const { return components * point_count; }
  bool HasExpectedSize() const
  {
    return storage != nullptr && storage->Size() == ExpectedSize();
  }
};

inline VolumeFluxCorrectionFieldBinding BindVolumeFluxCorrection(
  mfem::Vector& storage,
  int components,
  int dimension,
  int point_count)
{
  return {&storage, components, dimension, point_count};
}

inline StateVectorCorrectionFieldBinding BindStateVectorCorrection(
  mfem::Vector& storage,
  int components,
  int point_count)
{
  return {&storage, components, point_count};
}

static_assert(
  std::is_trivially_copyable<VolumeFluxCorrectionFieldBinding>::value,
  "volume correction binding must be trivially copyable");
static_assert(
  std::is_trivially_copyable<StateVectorCorrectionFieldBinding>::value,
  "state-vector correction binding must be trivially copyable");
static_assert(
  !std::is_same<VolumeFluxCorrectionFieldBinding,
                StateVectorCorrectionFieldBinding>::value,
  "volume tensors and state vectors must remain distinct public types");

/** All data dependencies of one point-kernel launch.
 *
 * Layouts are component-major within each point:
 * - state: `(state_component, point)`;
 * - state_gradient: `(state_component, direction, point)`;
 * - volume correction: `(direction, correction_component, point)`;
 * - normal/wall correction: `(correction_component, point)`.
 *
 * The typed auxiliary descriptors become trivially-copyable bindings at this
 * boundary; their value kind, topology, ownership, identity, capabilities, and
 * provenance remain available to host validation.
 */
struct RegularizationLaunchBuffers
{
  int point_count = 0;
  int dimension = 0;
  int state_components = 0;

  const mfem::Vector* state = nullptr;
  AuxiliaryFieldBinding auxiliary{};
  AuxiliaryFieldBinding trace{};
  const mfem::Vector* state_gradient = nullptr;
  const mfem::Vector* normals = nullptr;

  mfem::Vector* coefficient = nullptr;
  mfem::Vector* source = nullptr;
  VolumeFluxCorrectionFieldBinding volume_correction{};
  StateVectorCorrectionFieldBinding normal_correction{};
  StateVectorCorrectionFieldBinding wall_flux{};
  mfem::Vector* characteristic_bounds = nullptr;
  mfem::Vector* diagnostics = nullptr;
};

bool IsRuntimeCombinationSupported(const RegularizationSelection& selection);

ValidationResult ValidateRegularizationLaunch(
  const RegularizationSelection& selection,
  const RegularizationValidationContext& context,
  const RegularizationLaunchBuffers& buffers);

} // namespace regularization
} // namespace dg_igr

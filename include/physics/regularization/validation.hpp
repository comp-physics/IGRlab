#pragma once

#include "physics/regularization/auxiliary_field.hpp"
#include "physics/regularization/axes.hpp"

#include "mfem.hpp"

#include <cstdint>

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
  InvalidPointCount,
  MissingBuffer,
  InvalidBufferSize,
  InvalidFieldIdentity,
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

/** All data dependencies of one point-kernel launch. Inputs and outputs use
 * MFEM Vector/Memory storage; the scalar auxiliary and trace fields retain
 * their typed identity/provenance descriptors. */
struct RegularizationLaunchBuffers
{
  int point_count = 0;
  int dimension = 0;

  const mfem::Vector* state = nullptr;
  const ScalarAuxiliaryFieldDescriptor* auxiliary = nullptr;
  const TraceAuxiliaryFieldDescriptor* trace = nullptr;
  const mfem::Vector* gradient = nullptr;
  const mfem::Vector* normals = nullptr;

  mfem::Vector* coefficient = nullptr;
  mfem::Vector* source = nullptr;
  mfem::Vector* volume_correction = nullptr;
  mfem::Vector* normal_correction = nullptr;
  mfem::Vector* wall_flux = nullptr;
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

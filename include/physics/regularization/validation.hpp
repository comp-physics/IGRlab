#pragma once

#include "physics/regularization/auxiliary_field.hpp"
#include "physics/regularization/axes.hpp"
#include "physics/regularization/evaluated_field.hpp"

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
  UnsupportedCapability,
  BackendUnavailable,
  InvalidModelShape,
  InvalidDimension,
  InvalidComponentCount,
  InvalidPointCount,
  InvalidEvaluationSite,
  InvalidPointSetIdentity,
  MissingBuffer,
  InvalidBufferSize,
  InvalidBufferAliasing,
  InvalidFieldIdentity,
  InvalidFieldValueKind,
  InvalidFieldTopology,
  MissingFieldCapability,
  InvalidProvenanceStamp,
  StaleProvenance
};

struct ValidationResult
{
  ValidationCode code = ValidationCode::Valid;
  const char* message = "valid";

  constexpr bool Ok() const { return code == ValidationCode::Valid; }
};

struct ExpectedEvaluatedPointProducerGenerations
{
  std::uint64_t state = 0u;
  std::uint64_t conservative_state_gradient = 0u;
  std::uint64_t auxiliary = 0u;
  std::uint64_t trace = 0u;
  std::uint64_t normal = 0u;
  std::uint64_t base_characteristic_speed = 0u;
};

struct RegularizationValidationContext
{
  bool mfem_device_available = false;
  int expected_rk_stage = -1;
  std::uint64_t expected_state_generation = 0u;
  ExpectedEvaluatedPointProducerGenerations expected_producer_generations{};
};

bool IsRuntimeCombinationSupported(const RegularizationSelection& selection);

ValidationResult
ValidateRegularizationSelection(const RegularizationSelection& selection);

ValidationResult
ValidateAuxiliaryFieldBinding(const AuxiliaryFieldBinding& binding);

ValidationResult ValidateConfiguredRegularizationLaunch(
  ExecutionTargetKind execution_target,
  RegularizationOperationKind operation,
  EvaluatedPointFieldRequirements requirements,
  const RegularizationValidationContext& context,
  const EvaluatedPointBatch& batch,
  const RegularizationOperationOutputs& outputs);

} // namespace regularization
} // namespace dg_igr

#include "physics/regularization/validation.hpp"

#include <limits>

namespace dg_igr
{
namespace regularization
{
namespace
{

bool Known(ContinuumModelKind value)
{
  return value == ContinuumModelKind::ZeroFixture ||
         value == ContinuumModelKind::ConstantFixture;
}

bool Known(AuxiliaryDiscretizationKind value)
{
  return value == AuxiliaryDiscretizationKind::ElementL2 ||
         value == AuxiliaryDiscretizationKind::TraceL2 ||
         value == AuxiliaryDiscretizationKind::HybridizedTrace;
}

bool Known(AuxiliarySolutionKind value)
{
  return value == AuxiliarySolutionKind::PointwiseEvaluation ||
         value == AuxiliarySolutionKind::LocalElimination ||
         value == AuxiliarySolutionKind::GlobalLinearSolve;
}

bool Known(SpatialBackendKind value)
{
  return value == SpatialBackendKind::DG || value == SpatialBackendKind::DGSEM ||
         value == SpatialBackendKind::HDG;
}

bool Known(ExecutionTargetKind value)
{
  return value == ExecutionTargetKind::MFEMCpu ||
         value == ExecutionTargetKind::MFEMDevice;
}

bool Known(TimeTreatmentKind value)
{
  return value == TimeTreatmentKind::FrozenRKStage ||
         value == TimeTreatmentKind::LaggedStep ||
         value == TimeTreatmentKind::FullyCoupled;
}

bool Known(EntropyTreatmentKind value)
{
  return value == EntropyTreatmentKind::Neutral ||
         value == EntropyTreatmentKind::EntropyConservative ||
         value == EntropyTreatmentKind::EntropyStable;
}

bool Known(AuxiliaryValueKind value)
{
  return value == AuxiliaryValueKind::Scalar ||
         value == AuxiliaryValueKind::Vector ||
         value == AuxiliaryValueKind::Tensor;
}

bool HasSize(const mfem::Vector* vector, int expected)
{
  return vector != nullptr && expected >= 0 && vector->Size() == expected;
}

bool ProductFits(int first, int second, int third, int& product)
{
  if (first < 0 || second < 0 || third < 0)
  {
    return false;
  }
  const int maximum = std::numeric_limits<int>::max();
  int value = first;
  if (second != 0 && value > maximum / second)
  {
    return false;
  }
  value *= second;
  if (third != 0 && value > maximum / third)
  {
    return false;
  }
  value *= third;
  product = value;
  return true;
}

bool ExpectedValueComponents(AuxiliaryValueKind value_kind,
                             int dimension,
                             int& components)
{
  switch (value_kind)
  {
    case AuxiliaryValueKind::Scalar:
      components = 1;
      return true;
    case AuxiliaryValueKind::Vector:
      components = dimension;
      return true;
    case AuxiliaryValueKind::Tensor:
      return ProductFits(dimension, dimension, 1, components);
  }
  return false;
}

bool HasForbiddenVectorAliasing(const RegularizationLaunchBuffers& buffers)
{
  const mfem::Vector* const inputs[] = {buffers.state,
                                        buffers.auxiliary.storage,
                                        buffers.trace.storage,
                                        buffers.state_gradient,
                                        buffers.normals};
  mfem::Vector* const outputs[] = {buffers.coefficient,
                                   buffers.source,
                                   buffers.volume_correction.storage,
                                   buffers.normal_correction.storage,
                                   buffers.wall_flux.storage,
                                   buffers.characteristic_bounds,
                                   buffers.diagnostics};
  constexpr int input_count = static_cast<int>(sizeof(inputs) / sizeof(inputs[0]));
  constexpr int output_count =
    static_cast<int>(sizeof(outputs) / sizeof(outputs[0]));
  for (int output = 0; output < output_count; ++output)
  {
    for (int input = 0; input < input_count; ++input)
    {
      if (outputs[output] == inputs[input])
      {
        return true;
      }
    }
    for (int previous = 0; previous < output; ++previous)
    {
      if (outputs[output] == outputs[previous])
      {
        return true;
      }
    }
  }
  return false;
}

ValidationResult ValidateIdentityAndCapabilities(
  const AuxiliaryFieldBinding& auxiliary,
  const AuxiliaryFieldBinding& trace,
  ExecutionTargetKind execution)
{
  if (auxiliary.topology != AuxiliaryFieldTopology::Volume ||
      trace.topology != AuxiliaryFieldTopology::Trace)
  {
    return {ValidationCode::InvalidFieldTopology,
            "launch requires a volume auxiliary and a trace field"};
  }

  if (auxiliary.identity.semantic_id == 0u ||
      trace.identity.semantic_id == 0u ||
      auxiliary.identity.semantic_id == trace.identity.semantic_id)
  {
    return {ValidationCode::InvalidFieldIdentity,
            "auxiliary and trace fields require distinct nonzero semantic identities"};
  }

  if (!auxiliary.capabilities.Has(AuxiliaryCapability::StageLocal) ||
      !trace.capabilities.Has(AuxiliaryCapability::StageLocal) ||
      !trace.capabilities.Has(AuxiliaryCapability::TraceCoupled))
  {
    return {ValidationCode::MissingFieldCapability,
            "launch fields lack required stage/trace capabilities"};
  }

  if ((auxiliary.capabilities.Has(AuxiliaryCapability::Restartable) &&
       auxiliary.identity.restart_id == 0u) ||
      (trace.capabilities.Has(AuxiliaryCapability::Restartable) &&
       trace.identity.restart_id == 0u) ||
      (auxiliary.capabilities.Has(AuxiliaryCapability::OutputVisible) &&
       auxiliary.identity.output_id == 0u) ||
      (trace.capabilities.Has(AuxiliaryCapability::OutputVisible) &&
       trace.identity.output_id == 0u))
  {
    return {ValidationCode::InvalidFieldIdentity,
            "restart/output capability requires a stable identity"};
  }

  if (auxiliary.capabilities.Has(AuxiliaryCapability::Restartable) &&
      trace.capabilities.Has(AuxiliaryCapability::Restartable) &&
      auxiliary.identity.restart_id == trace.identity.restart_id)
  {
    return {ValidationCode::InvalidFieldIdentity,
            "restartable auxiliary and trace fields require distinct restart identities"};
  }

  if (auxiliary.capabilities.Has(AuxiliaryCapability::OutputVisible) &&
      trace.capabilities.Has(AuxiliaryCapability::OutputVisible) &&
      auxiliary.identity.output_id == trace.identity.output_id)
  {
    return {ValidationCode::InvalidFieldIdentity,
            "output-visible auxiliary and trace fields require distinct output identities"};
  }

  if (execution == ExecutionTargetKind::MFEMDevice &&
      (!auxiliary.capabilities.Has(AuxiliaryCapability::DeviceAccessible) ||
       !trace.capabilities.Has(AuxiliaryCapability::DeviceAccessible)))
  {
    return {ValidationCode::MissingFieldCapability,
            "device execution requires device-accessible fields"};
  }

  return {};
}

} // namespace

bool IsRuntimeCombinationSupported(const RegularizationSelection& selection)
{
  const RegularizationAxes& axes = selection.axes;
  const RegularizationCompatibility& compatibility = selection.compatibility;

  if (!Known(axes.continuum_model) ||
      !Known(axes.auxiliary_discretization) ||
      !Known(axes.auxiliary_solution) || !Known(axes.backend.spatial) ||
      !Known(axes.backend.execution) || !Known(compatibility.time) ||
      !Known(compatibility.entropy))
  {
    return false;
  }

  if (compatibility.time != TimeTreatmentKind::FrozenRKStage ||
      compatibility.entropy != EntropyTreatmentKind::Neutral)
  {
    return false;
  }

  if (axes.auxiliary_discretization ==
        AuxiliaryDiscretizationKind::ElementL2 &&
      axes.auxiliary_solution == AuxiliarySolutionKind::PointwiseEvaluation)
  {
    return true;
  }

  if (axes.continuum_model != ContinuumModelKind::ConstantFixture)
  {
    return false;
  }

  const bool local_trace =
    axes.auxiliary_discretization == AuxiliaryDiscretizationKind::TraceL2 &&
    axes.auxiliary_solution == AuxiliarySolutionKind::LocalElimination &&
    (axes.backend.spatial == SpatialBackendKind::DG ||
     axes.backend.spatial == SpatialBackendKind::DGSEM);
  const bool global_hybrid =
    axes.auxiliary_discretization ==
      AuxiliaryDiscretizationKind::HybridizedTrace &&
    axes.auxiliary_solution == AuxiliarySolutionKind::GlobalLinearSolve &&
    axes.backend.spatial == SpatialBackendKind::HDG;
  return local_trace || global_hybrid;
}

ValidationResult ValidateRegularizationSelection(
  const RegularizationSelection& selection)
{
  const RegularizationAxes& axes = selection.axes;
  const RegularizationCompatibility& compatibility = selection.compatibility;
  if (!Known(axes.continuum_model) ||
      !Known(axes.auxiliary_discretization) ||
      !Known(axes.auxiliary_solution) || !Known(axes.backend.spatial) ||
      !Known(axes.backend.execution) || !Known(compatibility.time) ||
      !Known(compatibility.entropy))
  {
    return {ValidationCode::UnknownAxisValue,
            "selection contains an unknown axis value"};
  }

  if (!IsRuntimeCombinationSupported(selection))
  {
    return {ValidationCode::UnsupportedCombination,
            "model/discretization/solve/spatial/execution/time/entropy combination is unsupported"};
  }
  return {};
}

ValidationResult ValidateRegularizationLaunch(
  const RegularizationSelection& selection,
  const RegularizationValidationContext& context,
  const RegularizationLaunchBuffers& buffers)
{
  const ValidationResult selection_validation =
    ValidateRegularizationSelection(selection);
  if (!selection_validation.Ok())
  {
    return selection_validation;
  }
  return ValidateConfiguredRegularizationLaunch(
    selection.axes.backend.execution, context, buffers);
}

ValidationResult ValidateConfiguredRegularizationLaunch(
  ExecutionTargetKind execution_target,
  const RegularizationValidationContext& context,
  const RegularizationLaunchBuffers& buffers)
{
  if (!Known(execution_target))
  {
    return {ValidationCode::UnknownAxisValue,
            "configured launch contains an unknown execution target"};
  }

  if (execution_target == ExecutionTargetKind::MFEMDevice &&
      !context.mfem_device_available)
  {
    return {ValidationCode::BackendUnavailable,
            "MFEM device execution target is unavailable"};
  }

  if (context.expected_rk_stage < 0 ||
      context.expected_state_generation == 0u)
  {
    return {ValidationCode::InvalidProvenanceStamp,
            "launch validation requires a stamped RK stage and state generation"};
  }

  if (buffers.dimension < 1 || buffers.dimension > 3)
  {
    return {ValidationCode::InvalidDimension,
            "point-kernel dimension must be in [1,3]"};
  }
  if (buffers.state_components <= 0 ||
      buffers.volume_correction.components <= 0 ||
      buffers.normal_correction.components <= 0 ||
      buffers.wall_flux.components <= 0 ||
      buffers.auxiliary.components <= 0 || buffers.trace.components <= 0)
  {
    return {ValidationCode::InvalidComponentCount,
            "state, auxiliary, trace, and correction component counts must be positive"};
  }
  if (buffers.point_count <= 0)
  {
    return {ValidationCode::InvalidPointCount,
            "point-kernel launch requires at least one point"};
  }

  if (!Known(buffers.auxiliary.value_kind) ||
      !Known(buffers.trace.value_kind))
  {
    return {ValidationCode::InvalidFieldValueKind,
            "auxiliary and trace fields require a known value kind"};
  }

  int expected_auxiliary_components = 0;
  int expected_trace_components = 0;
  if (!ExpectedValueComponents(buffers.auxiliary.value_kind,
                               buffers.dimension,
                               expected_auxiliary_components) ||
      !ExpectedValueComponents(buffers.trace.value_kind,
                               buffers.dimension,
                               expected_trace_components))
  {
    return {ValidationCode::InvalidComponentCount,
            "auxiliary or trace value-kind extent exceeds MFEM Vector limits"};
  }
  if (buffers.auxiliary.components != expected_auxiliary_components ||
      buffers.trace.components != expected_trace_components)
  {
    return {ValidationCode::InvalidComponentCount,
            "auxiliary and trace component extents must match their value kinds"};
  }

  if (buffers.volume_correction.components != buffers.state_components ||
      buffers.normal_correction.components !=
        buffers.volume_correction.components ||
      buffers.wall_flux.components != buffers.volume_correction.components)
  {
    return {ValidationCode::InvalidBufferSize,
            "volume, normal, and wall correction extents must match the state component extent"};
  }

  if (buffers.state == nullptr || buffers.auxiliary.storage == nullptr ||
      buffers.trace.storage == nullptr || buffers.state_gradient == nullptr ||
      buffers.normals == nullptr || buffers.coefficient == nullptr ||
      buffers.source == nullptr ||
      buffers.volume_correction.storage == nullptr ||
      buffers.normal_correction.storage == nullptr ||
      buffers.wall_flux.storage == nullptr ||
      buffers.characteristic_bounds == nullptr ||
      buffers.diagnostics == nullptr)
  {
    return {ValidationCode::MissingBuffer,
            "regularization launch is missing an input or output buffer"};
  }

  if (HasForbiddenVectorAliasing(buffers))
  {
    return {ValidationCode::InvalidBufferAliasing,
            "writable outputs must not alias inputs or other writable outputs"};
  }

  const int points = buffers.point_count;
  const int dimension = buffers.dimension;
  const int correction_components = buffers.volume_correction.components;
  int state_size = 0;
  int gradient_size = 0;
  int coefficient_size = 0;
  int trace_size = 0;
  int volume_size = 0;
  int correction_size = 0;
  int normal_input_size = 0;
  int bounds_size = 0;
  int diagnostics_size = 0;
  if (!ProductFits(buffers.state_components, points, 1, state_size) ||
      !ProductFits(buffers.state_components,
                   dimension,
                   points,
                   gradient_size) ||
      !ProductFits(buffers.auxiliary.components,
                   points,
                   1,
                   coefficient_size) ||
      !ProductFits(buffers.trace.components,
                   points,
                   1,
                   trace_size) ||
      !ProductFits(correction_components,
                   dimension,
                   points,
                   volume_size) ||
      !ProductFits(correction_components,
                   points,
                   1,
                   correction_size) ||
      !ProductFits(dimension, points, 1, normal_input_size) ||
      !ProductFits(2, points, 1, bounds_size) ||
      !ProductFits(3, points, 1, diagnostics_size))
  {
    return {ValidationCode::InvalidComponentCount,
            "regularization launch shape exceeds MFEM Vector limits"};
  }

  if (!HasSize(buffers.state, state_size) ||
      buffers.auxiliary.entity_count != points ||
      buffers.auxiliary.dimension != dimension ||
      !HasSize(buffers.auxiliary.storage, coefficient_size) ||
      buffers.trace.entity_count != points ||
      buffers.trace.dimension != dimension ||
      !HasSize(buffers.trace.storage, trace_size) ||
      !HasSize(buffers.state_gradient, gradient_size) ||
      !HasSize(buffers.normals, normal_input_size) ||
      !HasSize(buffers.coefficient, coefficient_size) ||
      !HasSize(buffers.source, coefficient_size) ||
      buffers.volume_correction.dimension != dimension ||
      buffers.volume_correction.point_count != points ||
      buffers.normal_correction.components != correction_components ||
      buffers.normal_correction.point_count != points ||
      buffers.wall_flux.components != correction_components ||
      buffers.wall_flux.point_count != points ||
      !HasSize(buffers.volume_correction.storage, volume_size) ||
      !HasSize(buffers.normal_correction.storage, correction_size) ||
      !HasSize(buffers.wall_flux.storage, correction_size) ||
      !HasSize(buffers.characteristic_bounds, bounds_size) ||
      !HasSize(buffers.diagnostics, diagnostics_size))
  {
    return {ValidationCode::InvalidBufferSize,
            "regularization launch buffer size does not match its typed shape"};
  }

  const ValidationResult field_validation = ValidateIdentityAndCapabilities(
    buffers.auxiliary, buffers.trace, execution_target);
  if (!field_validation.Ok())
  {
    return field_validation;
  }

  const AuxiliaryFieldProvenance expected{
    context.expected_rk_stage, context.expected_state_generation, 0u};
  if (!SameStageState(buffers.auxiliary.provenance, expected) ||
      !SameStageState(buffers.trace.provenance, expected))
  {
    return {ValidationCode::StaleProvenance,
            "auxiliary field provenance does not match the RK-stage state"};
  }

  return {};
}

} // namespace regularization
} // namespace dg_igr

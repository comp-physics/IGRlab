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
  const long long value = static_cast<long long>(first) * second * third;
  if (value > std::numeric_limits<int>::max())
  {
    return false;
  }
  product = static_cast<int>(value);
  return true;
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
      trace.identity.semantic_id == 0u || auxiliary.identity == trace.identity)
  {
    return {ValidationCode::InvalidFieldIdentity,
            "auxiliary and trace fields require distinct nonzero identities"};
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

ValidationResult ValidateRegularizationLaunch(
  const RegularizationSelection& selection,
  const RegularizationValidationContext& context,
  const RegularizationLaunchBuffers& buffers)
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

  if (axes.backend.execution == ExecutionTargetKind::MFEMDevice &&
      !context.mfem_device_available)
  {
    return {ValidationCode::BackendUnavailable,
            "MFEM device execution target is unavailable"};
  }

  if (buffers.dimension < 1 || buffers.dimension > 3)
  {
    return {ValidationCode::InvalidDimension,
            "point-kernel dimension must be in [1,3]"};
  }
  if (buffers.state_components <= 0 ||
      buffers.volume_correction.components <= 0 ||
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

  const int points = buffers.point_count;
  const int dimension = buffers.dimension;
  const int correction_components = buffers.volume_correction.components;
  int state_size = 0;
  int gradient_size = 0;
  int coefficient_size = 0;
  int volume_size = 0;
  int correction_size = 0;
  if (!ProductFits(buffers.state_components, points, 1, state_size) ||
      !ProductFits(buffers.state_components,
                   dimension,
                   points,
                   gradient_size) ||
      !ProductFits(buffers.auxiliary.components,
                   points,
                   1,
                   coefficient_size) ||
      !ProductFits(correction_components,
                   dimension,
                   points,
                   volume_size) ||
      !ProductFits(correction_components,
                   points,
                   1,
                   correction_size))
  {
    return {ValidationCode::InvalidComponentCount,
            "regularization launch shape exceeds MFEM Vector limits"};
  }

  if (!HasSize(buffers.state, state_size) ||
      buffers.auxiliary.entity_count != points ||
      buffers.auxiliary.dimension != dimension ||
      !buffers.auxiliary.HasExpectedSize() ||
      buffers.trace.entity_count != points ||
      buffers.trace.dimension != dimension ||
      !buffers.trace.HasExpectedSize() ||
      !HasSize(buffers.state_gradient, gradient_size) ||
      !HasSize(buffers.normals, dimension * points) ||
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
      !HasSize(buffers.characteristic_bounds, 2 * points) ||
      !HasSize(buffers.diagnostics, 3 * points))
  {
    return {ValidationCode::InvalidBufferSize,
            "regularization launch buffer size does not match its typed shape"};
  }

  const ValidationResult field_validation = ValidateIdentityAndCapabilities(
    buffers.auxiliary, buffers.trace, axes.backend.execution);
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

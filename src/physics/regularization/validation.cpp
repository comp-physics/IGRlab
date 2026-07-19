#include "physics/regularization/validation.hpp"

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

bool Known(ExecutionBackendKind value)
{
  return value == ExecutionBackendKind::MFEMCpu ||
         value == ExecutionBackendKind::MFEMDevice;
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
  return vector != nullptr && vector->Size() == expected;
}

ValidationResult ValidateIdentityAndCapabilities(
  const ScalarAuxiliaryFieldDescriptor& auxiliary,
  const TraceAuxiliaryFieldDescriptor& trace,
  ExecutionBackendKind backend)
{
  const AuxiliaryFieldIdentity auxiliary_id = auxiliary.Identity();
  const AuxiliaryFieldIdentity trace_id = trace.Identity();
  if (auxiliary_id.semantic_id == 0u || trace_id.semantic_id == 0u ||
      auxiliary_id == trace_id)
  {
    return {ValidationCode::InvalidFieldIdentity,
            "auxiliary and trace fields require distinct nonzero identities"};
  }

  const AuxiliaryFieldCapabilities auxiliary_caps = auxiliary.Capabilities();
  const AuxiliaryFieldCapabilities trace_caps = trace.Capabilities();
  if (!auxiliary_caps.Has(AuxiliaryCapability::StageLocal) ||
      !trace_caps.Has(AuxiliaryCapability::StageLocal) ||
      !trace_caps.Has(AuxiliaryCapability::TraceCoupled))
  {
    return {ValidationCode::MissingFieldCapability,
            "launch fields lack required stage/trace capabilities"};
  }

  if ((auxiliary_caps.Has(AuxiliaryCapability::Restartable) &&
       auxiliary_id.restart_id == 0u) ||
      (trace_caps.Has(AuxiliaryCapability::Restartable) &&
       trace_id.restart_id == 0u) ||
      (auxiliary_caps.Has(AuxiliaryCapability::OutputVisible) &&
       auxiliary_id.output_id == 0u) ||
      (trace_caps.Has(AuxiliaryCapability::OutputVisible) &&
       trace_id.output_id == 0u))
  {
    return {ValidationCode::InvalidFieldIdentity,
            "restart/output capability requires a stable identity"};
  }

  if (backend == ExecutionBackendKind::MFEMDevice &&
      (!auxiliary_caps.Has(AuxiliaryCapability::DeviceAccessible) ||
       !trace_caps.Has(AuxiliaryCapability::DeviceAccessible)))
  {
    return {ValidationCode::MissingFieldCapability,
            "device backend requires device-accessible fields"};
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
      !Known(axes.auxiliary_solution) || !Known(axes.execution_backend) ||
      !Known(compatibility.time) || !Known(compatibility.entropy))
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
      axes.auxiliary_solution ==
        AuxiliarySolutionKind::PointwiseEvaluation)
  {
    return true;
  }

  if (axes.continuum_model != ContinuumModelKind::ConstantFixture ||
      axes.execution_backend != ExecutionBackendKind::MFEMCpu)
  {
    return false;
  }

  const bool local_trace =
    axes.auxiliary_discretization ==
      AuxiliaryDiscretizationKind::TraceL2 &&
    axes.auxiliary_solution == AuxiliarySolutionKind::LocalElimination;
  const bool global_hybrid =
    axes.auxiliary_discretization ==
      AuxiliaryDiscretizationKind::HybridizedTrace &&
    axes.auxiliary_solution == AuxiliarySolutionKind::GlobalLinearSolve;
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
      !Known(axes.auxiliary_solution) || !Known(axes.execution_backend) ||
      !Known(compatibility.time) || !Known(compatibility.entropy))
  {
    return {ValidationCode::UnknownAxisValue,
            "selection contains an unknown axis value"};
  }

  if (!IsRuntimeCombinationSupported(selection))
  {
    return {ValidationCode::UnsupportedCombination,
            "model/discretization/solve/backend/time/entropy combination is unsupported"};
  }

  if (axes.execution_backend == ExecutionBackendKind::MFEMDevice &&
      !context.mfem_device_available)
  {
    return {ValidationCode::BackendUnavailable,
            "MFEM device backend is unavailable"};
  }

  if (buffers.dimension < 1 || buffers.dimension > 3)
  {
    return {ValidationCode::InvalidDimension,
            "point-kernel dimension must be in [1,3]"};
  }
  if (buffers.point_count <= 0)
  {
    return {ValidationCode::InvalidPointCount,
            "point-kernel launch requires at least one point"};
  }

  if (buffers.state == nullptr || buffers.auxiliary == nullptr ||
      buffers.trace == nullptr || buffers.gradient == nullptr ||
      buffers.normals == nullptr || buffers.coefficient == nullptr ||
      buffers.source == nullptr || buffers.volume_correction == nullptr ||
      buffers.normal_correction == nullptr || buffers.wall_flux == nullptr ||
      buffers.characteristic_bounds == nullptr ||
      buffers.diagnostics == nullptr)
  {
    return {ValidationCode::MissingBuffer,
            "regularization launch is missing an input or output buffer"};
  }

  const int points = buffers.point_count;
  const int dimension = buffers.dimension;
  if (!HasSize(buffers.state, points) ||
      buffers.auxiliary->EntityCount() != points ||
      buffers.auxiliary->Dimension() != dimension ||
      !buffers.auxiliary->HasExpectedSize() ||
      buffers.trace->EntityCount() != points ||
      buffers.trace->Dimension() != dimension ||
      !buffers.trace->HasExpectedSize() ||
      !HasSize(buffers.gradient, dimension * points) ||
      !HasSize(buffers.normals, dimension * points) ||
      !HasSize(buffers.coefficient, points) ||
      !HasSize(buffers.source, points) ||
      !HasSize(buffers.volume_correction, dimension * points) ||
      !HasSize(buffers.normal_correction, points) ||
      !HasSize(buffers.wall_flux, points) ||
      !HasSize(buffers.characteristic_bounds, 2 * points) ||
      !HasSize(buffers.diagnostics, 3 * points))
  {
    return {ValidationCode::InvalidBufferSize,
            "regularization launch buffer size does not match its typed shape"};
  }

  const ValidationResult field_validation = ValidateIdentityAndCapabilities(
    *buffers.auxiliary, *buffers.trace, axes.execution_backend);
  if (!field_validation.Ok())
  {
    return field_validation;
  }

  const AuxiliaryFieldProvenance expected{
    context.expected_rk_stage, context.expected_state_generation, 0u};
  if (!SameStageState(buffers.auxiliary->Provenance(), expected) ||
      !SameStageState(buffers.trace->Provenance(), expected))
  {
    return {ValidationCode::StaleProvenance,
            "auxiliary field provenance does not match the RK-stage state"};
  }

  return {};
}

} // namespace regularization
} // namespace dg_igr

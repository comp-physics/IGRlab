#include "igrlab/compat.hpp"
#include "physics/regularization/combination.hpp"
#include "physics/regularization/validation.hpp"

#include <limits>

namespace igrlab
{
namespace regularization
{
namespace
{

bool Known(ContinuumModelKind value)
{
  return value == ContinuumModelKind::ZeroFixture ||
         value == ContinuumModelKind::ConstantFixture ||
         value == ContinuumModelKind::BrokenDGContractProbe;
}

bool Known(GradientTreatmentKind value)
{
  return value == GradientTreatmentKind::None ||
         value == GradientTreatmentKind::ConservativeStateBrokenDG;
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
  return value == SpatialBackendKind::DG ||
         value == SpatialBackendKind::DGSEM || value == SpatialBackendKind::HDG;
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

bool Known(RegularizationOperationKind value)
{
  return value == RegularizationOperationKind::AuxiliaryCoefficient ||
         value == RegularizationOperationKind::AuxiliarySource ||
         value == RegularizationOperationKind::VolumeCorrection ||
         value == RegularizationOperationKind::NormalCorrection ||
         value == RegularizationOperationKind::WallLaw ||
         value == RegularizationOperationKind::CharacteristicBounds ||
         value == RegularizationOperationKind::Diagnostics;
}

bool Known(EvaluationSiteKind value)
{
  return value == EvaluationSiteKind::Volume ||
         value == EvaluationSiteKind::Face || value == EvaluationSiteKind::Wall;
}

bool Known(AuxiliaryRestartKind value)
{
  return value == AuxiliaryRestartKind::None ||
         value == AuxiliaryRestartKind::Serialized ||
         value == AuxiliaryRestartKind::ColdRecomputed;
}

bool Known(AuxiliaryValueKind value)
{
  return value == AuxiliaryValueKind::Scalar ||
         value == AuxiliaryValueKind::Vector ||
         value == AuxiliaryValueKind::Tensor;
}

bool Known(AuxiliaryFieldTopology value)
{
  return value == AuxiliaryFieldTopology::Volume ||
         value == AuxiliaryFieldTopology::Trace;
}

bool ProductFits(int first, int second, int& product)
{
  if (first < 0 || second < 0)
  {
    return false;
  }
  if (second != 0 && first > std::numeric_limits<int>::max() / second)
  {
    return false;
  }
  product = first * second;
  return true;
}

EvaluationSiteKind RequiredSite(RegularizationOperationKind operation)
{
  switch (operation)
  {
  case RegularizationOperationKind::AuxiliaryCoefficient:
  case RegularizationOperationKind::AuxiliarySource:
  case RegularizationOperationKind::VolumeCorrection:
  case RegularizationOperationKind::Diagnostics:
    return EvaluationSiteKind::Volume;
  case RegularizationOperationKind::NormalCorrection:
  case RegularizationOperationKind::CharacteristicBounds:
    return EvaluationSiteKind::Face;
  case RegularizationOperationKind::WallLaw:
    return EvaluationSiteKind::Wall;
  }
  return EvaluationSiteKind::Volume;
}

const EvaluatedPointOutputBinding&
SelectedOutput(RegularizationOperationKind operation,
               const RegularizationOperationOutputs& outputs)
{
  switch (operation)
  {
  case RegularizationOperationKind::AuxiliaryCoefficient:
    return outputs.auxiliary_coefficient;
  case RegularizationOperationKind::AuxiliarySource:
    return outputs.auxiliary_source;
  case RegularizationOperationKind::VolumeCorrection:
    return outputs.volume_correction;
  case RegularizationOperationKind::NormalCorrection:
    return outputs.normal_correction;
  case RegularizationOperationKind::WallLaw:
    return outputs.wall_flux;
  case RegularizationOperationKind::CharacteristicBounds:
    return outputs.characteristic_bounds;
  case RegularizationOperationKind::Diagnostics:
    return outputs.diagnostics;
  }
  return outputs.auxiliary_coefficient;
}

int ExpectedOutputComponents(RegularizationOperationKind operation,
                             const EvaluatedPointBatch& batch,
                             bool& fits)
{
  fits = true;
  switch (operation)
  {
  case RegularizationOperationKind::AuxiliaryCoefficient:
  case RegularizationOperationKind::AuxiliarySource:
    return batch.auxiliary.ComponentExtent();
  case RegularizationOperationKind::VolumeCorrection:
  {
    int components = 0;
    fits =
      ProductFits(batch.state.ComponentExtent(), batch.dimension, components);
    return components;
  }
  case RegularizationOperationKind::NormalCorrection:
  case RegularizationOperationKind::WallLaw:
    return batch.state.ComponentExtent();
  case RegularizationOperationKind::CharacteristicBounds:
    return 2;
  case RegularizationOperationKind::Diagnostics:
    return 3;
  }
  fits = false;
  return 0;
}

ValidationResult ValidateRequiredField(const EvaluatedPointFieldBinding& field,
                                       const EvaluatedPointBatch& batch,
                                       std::uint64_t expected_producer_generation)
{
  const EvaluatedPointFieldMetadata& metadata = field.Metadata();
  if (!Known(metadata.site))
  {
    return {ValidationCode::InvalidEvaluationSite,
            "required evaluated field has an unknown site"};
  }
  if (metadata.point_count <= 0)
  {
    return {ValidationCode::InvalidPointCount,
            "required evaluated field has no evaluated points"};
  }
  if (metadata.component_extent <= 0)
  {
    return {ValidationCode::InvalidComponentCount,
            "required evaluated field has no components"};
  }
  if (metadata.dimension < 1 || metadata.dimension > 3)
  {
    return {ValidationCode::InvalidDimension,
            "required evaluated field has invalid dimension"};
  }
  if (metadata.point_set_id == 0u)
  {
    return {ValidationCode::InvalidPointSetIdentity,
            "required evaluated field needs a nonzero point-set identity"};
  }
  if (metadata.rk_stage < 0 || metadata.state_generation == 0u ||
      metadata.producer_generation == 0u ||
      expected_producer_generation == 0u)
  {
    return {
      ValidationCode::InvalidProvenanceStamp,
      "required evaluated field needs a nonnegative stage and nonzero "
      "state/producer generations"};
  }
  if (metadata.rk_stage != batch.rk_stage ||
      metadata.state_generation != batch.state_generation)
  {
    return {ValidationCode::StaleProvenance,
            "required evaluated field is stale for the selected batch"};
  }
  if (metadata.producer_generation != expected_producer_generation)
  {
    return {
      ValidationCode::StaleProvenance,
      "required evaluated field has a stale producer generation"};
  }
  if (metadata.site != batch.site)
  {
    return {ValidationCode::InvalidEvaluationSite,
            "required evaluated fields must use one evaluation site"};
  }
  if (metadata.point_count != batch.point_count)
  {
    return {ValidationCode::InvalidPointCount,
            "required evaluated fields must use one point count"};
  }
  if (metadata.point_set_id != batch.point_set_id)
  {
    return {ValidationCode::InvalidPointSetIdentity,
            "required evaluated fields must use one point-set identity"};
  }
  if (metadata.dimension != batch.dimension)
  {
    return {ValidationCode::InvalidDimension,
            "required evaluated fields must use one dimension"};
  }
  return {};
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag>
bool RuntimeEntropySupport(EntropyTreatmentKind entropy)
{
  switch (entropy)
  {
  case EntropyTreatmentKind::Neutral:
    return StaticCombinationSupport<ModelTag,
                                    DiscretizationTag,
                                    SolutionTag,
                                    BackendTag,
                                    GradientTag,
                                    TimeTag,
                                    EntropyNeutralTag>::value;
  case EntropyTreatmentKind::EntropyConservative:
    return StaticCombinationSupport<ModelTag,
                                    DiscretizationTag,
                                    SolutionTag,
                                    BackendTag,
                                    GradientTag,
                                    TimeTag,
                                    EntropyConservativeTag>::value;
  case EntropyTreatmentKind::EntropyStable:
    return StaticCombinationSupport<ModelTag,
                                    DiscretizationTag,
                                    SolutionTag,
                                    BackendTag,
                                    GradientTag,
                                    TimeTag,
                                    EntropyStableTag>::value;
  }
  return false;
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag>
bool RuntimeTimeSupport(const RegularizationCompatibility& compatibility)
{
  switch (compatibility.time)
  {
  case TimeTreatmentKind::FrozenRKStage:
    return RuntimeEntropySupport<ModelTag,
                                 DiscretizationTag,
                                 SolutionTag,
                                 BackendTag,
                                 GradientTag,
                                 FrozenRKStageTag>(compatibility.entropy);
  case TimeTreatmentKind::LaggedStep:
    return RuntimeEntropySupport<ModelTag,
                                 DiscretizationTag,
                                 SolutionTag,
                                 BackendTag,
                                 GradientTag,
                                 LaggedStepTag>(compatibility.entropy);
  case TimeTreatmentKind::FullyCoupled:
    return RuntimeEntropySupport<ModelTag,
                                 DiscretizationTag,
                                 SolutionTag,
                                 BackendTag,
                                 GradientTag,
                                 FullyCoupledTag>(compatibility.entropy);
  }
  return false;
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename SpatialTag,
          typename GradientTag>
bool RuntimeExecutionSupport(const RegularizationSelection& selection)
{
  switch (selection.axes.backend.execution)
  {
  case ExecutionTargetKind::MFEMCpu:
    return RuntimeTimeSupport<
      ModelTag,
      DiscretizationTag,
      SolutionTag,
      SpatialExecutionBackendTag<SpatialTag, MFEMCpuTargetTag>,
      GradientTag>(selection.compatibility);
  case ExecutionTargetKind::MFEMDevice:
    return RuntimeTimeSupport<
      ModelTag,
      DiscretizationTag,
      SolutionTag,
      SpatialExecutionBackendTag<SpatialTag, MFEMDeviceTargetTag>,
      GradientTag>(selection.compatibility);
  }
  return false;
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename GradientTag>
bool RuntimeSpatialSupport(const RegularizationSelection& selection)
{
  switch (selection.axes.backend.spatial)
  {
  case SpatialBackendKind::DG:
    return RuntimeExecutionSupport<ModelTag,
                                   DiscretizationTag,
                                   SolutionTag,
                                   DGSpatialBackendTag,
                                   GradientTag>(selection);
  case SpatialBackendKind::DGSEM:
    return RuntimeExecutionSupport<ModelTag,
                                   DiscretizationTag,
                                   SolutionTag,
                                   DGSEMSpatialBackendTag,
                                   GradientTag>(selection);
  case SpatialBackendKind::HDG:
    return RuntimeExecutionSupport<ModelTag,
                                   DiscretizationTag,
                                   SolutionTag,
                                   HDGSpatialBackendTag,
                                   GradientTag>(selection);
  }
  return false;
}

template <typename ModelTag, typename DiscretizationTag, typename GradientTag>
bool RuntimeSolutionSupport(const RegularizationSelection& selection)
{
  switch (selection.axes.auxiliary_solution)
  {
  case AuxiliarySolutionKind::PointwiseEvaluation:
    return RuntimeSpatialSupport<ModelTag,
                                 DiscretizationTag,
                                 PointwiseEvaluationTag,
                                 GradientTag>(selection);
  case AuxiliarySolutionKind::LocalElimination:
    return RuntimeSpatialSupport<ModelTag,
                                 DiscretizationTag,
                                 LocalEliminationTag,
                                 GradientTag>(selection);
  case AuxiliarySolutionKind::GlobalLinearSolve:
    return RuntimeSpatialSupport<ModelTag,
                                 DiscretizationTag,
                                 GlobalLinearSolveTag,
                                 GradientTag>(selection);
  }
  return false;
}

template <typename ModelTag, typename GradientTag>
bool RuntimeDiscretizationSupport(const RegularizationSelection& selection)
{
  switch (selection.axes.auxiliary_discretization)
  {
  case AuxiliaryDiscretizationKind::ElementL2:
    return RuntimeSolutionSupport<ModelTag, ElementL2Tag, GradientTag>(
      selection);
  case AuxiliaryDiscretizationKind::TraceL2:
    return RuntimeSolutionSupport<ModelTag, TraceL2Tag, GradientTag>(selection);
  case AuxiliaryDiscretizationKind::HybridizedTrace:
    return RuntimeSolutionSupport<ModelTag, HybridizedTraceTag, GradientTag>(
      selection);
  }
  return false;
}

template <typename ModelTag>
bool RuntimeGradientSupport(const RegularizationSelection& selection)
{
  switch (selection.axes.gradient_treatment)
  {
  case GradientTreatmentKind::None:
    return RuntimeDiscretizationSupport<ModelTag, NoneGradientTreatmentTag>(
      selection);
  case GradientTreatmentKind::ConservativeStateBrokenDG:
    return RuntimeDiscretizationSupport<ModelTag, ConservativeStateBrokenDGTag>(
      selection);
  }
  return false;
}

bool RuntimeModelSupport(const RegularizationSelection& selection)
{
  switch (selection.axes.continuum_model)
  {
  case ContinuumModelKind::ZeroFixture:
    return RuntimeGradientSupport<ZeroFixtureModelTag>(selection);
  case ContinuumModelKind::ConstantFixture:
    return RuntimeGradientSupport<ConstantFixtureModelTag>(selection);
  case ContinuumModelKind::BrokenDGContractProbe:
    return RuntimeGradientSupport<BrokenDGContractProbeModelTag>(selection);
  }
  return false;
}

} // namespace

bool IsRuntimeCombinationSupported(const RegularizationSelection& selection)
{
  const RegularizationAxes& axes = selection.axes;
  const RegularizationCompatibility& compatibility = selection.compatibility;
  if (!Known(axes.continuum_model) || !Known(axes.gradient_treatment) ||
      !Known(axes.auxiliary_discretization) ||
      !Known(axes.auxiliary_solution) || !Known(axes.backend.spatial) ||
      !Known(axes.backend.execution) || !Known(compatibility.time) ||
      !Known(compatibility.entropy))
  {
    return false;
  }
  return RuntimeModelSupport(selection);
}

ValidationResult
ValidateRegularizationSelection(const RegularizationSelection& selection)
{
  const RegularizationAxes& axes = selection.axes;
  const RegularizationCompatibility& compatibility = selection.compatibility;
  if (!Known(axes.continuum_model) || !Known(axes.gradient_treatment) ||
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
            "model/gradient/discretization/solve/backend/time/entropy "
            "combination is unsupported"};
  }
  return {};
}

ValidationResult
ValidateAuxiliaryFieldBinding(const AuxiliaryFieldBinding& binding)
{
  if (!Known(binding.value_kind))
  {
    return {ValidationCode::InvalidFieldValueKind,
            "auxiliary field has an unknown value kind"};
  }
  if (!Known(binding.topology))
  {
    return {ValidationCode::InvalidFieldTopology,
            "auxiliary field has an unknown topology"};
  }
  if (!Known(binding.restart_kind))
  {
    return {ValidationCode::UnknownAxisValue,
            "auxiliary field has an unknown restart kind"};
  }
  if (binding.identity.semantic_id == 0u)
  {
    return {ValidationCode::InvalidFieldIdentity,
            "auxiliary field requires a semantic identity"};
  }
  if (binding.restart_kind == AuxiliaryRestartKind::Serialized &&
      binding.identity.restart_id == 0u)
  {
    return {ValidationCode::InvalidFieldIdentity,
            "serialized auxiliary field requires a restart identity"};
  }
  if (binding.restart_kind != AuxiliaryRestartKind::Serialized &&
      binding.identity.restart_id != 0u)
  {
    return {
      ValidationCode::InvalidFieldIdentity,
      "None/ColdRecomputed auxiliary fields require zero restart identity"};
  }
  if (binding.capabilities.Has(AuxiliaryCapability::OutputVisible) &&
      binding.identity.output_id == 0u)
  {
    return {ValidationCode::InvalidFieldIdentity,
            "output-visible auxiliary field requires an output identity"};
  }
  if (binding.provenance.rk_stage < 0 ||
      binding.provenance.state_generation == 0u ||
      binding.provenance.producer_generation == 0u)
  {
    return {
      ValidationCode::InvalidProvenanceStamp,
      "auxiliary field requires a nonnegative stage and nonzero "
      "state/producer generations"};
  }
  return {};
}

ValidationResult ValidateConfiguredRegularizationLaunch(
  ExecutionTargetKind execution_target,
  RegularizationOperationKind operation,
  EvaluatedPointFieldRequirements requirements,
  const RegularizationValidationContext& context,
  const EvaluatedPointBatch& batch,
  const RegularizationOperationOutputs& outputs)
{
  if (!Known(execution_target) || !Known(operation))
  {
    return {ValidationCode::UnknownAxisValue,
            "configured launch contains an unknown target or operation"};
  }
  if (execution_target == ExecutionTargetKind::MFEMDevice &&
      !context.mfem_device_available)
  {
    return {ValidationCode::BackendUnavailable,
            "MFEM device execution target is unavailable"};
  }
  if (context.expected_rk_stage < 0 ||
      context.expected_state_generation == 0u || batch.rk_stage < 0 ||
      batch.state_generation == 0u)
  {
    return {
      ValidationCode::InvalidProvenanceStamp,
      "launch and batch require a nonnegative stage and nonzero state "
      "generation"};
  }
  if (batch.rk_stage != context.expected_rk_stage ||
      batch.state_generation != context.expected_state_generation)
  {
    return {ValidationCode::StaleProvenance,
            "evaluated batch is stale for the requested stage state"};
  }
  if (batch.dimension < 1 || batch.dimension > 3)
  {
    return {ValidationCode::InvalidDimension,
            "evaluated batch dimension must be in [1,3]"};
  }
  if (batch.point_count <= 0)
  {
    return {ValidationCode::InvalidPointCount,
            "operation launch requires at least one evaluated point"};
  }
  if (!Known(batch.site) || batch.site != RequiredSite(operation))
  {
    return {ValidationCode::InvalidEvaluationSite,
            "operation uses the wrong evaluated site"};
  }
  if (batch.point_set_id == 0u)
  {
    return {ValidationCode::InvalidPointSetIdentity,
            "evaluated batch requires a nonzero point-set identity"};
  }

  const EvaluatedPointFieldBinding* const fields[] = {
    &batch.state,
    &batch.conservative_state_gradient,
    &batch.auxiliary,
    &batch.trace,
    &batch.normal,
    &batch.base_characteristic_speed};
  const bool required[] = {requirements.state,
                           requirements.conservative_state_gradient,
                           requirements.auxiliary,
                           requirements.trace,
                           requirements.normal,
                           requirements.base_characteristic_speed};
  const std::uint64_t expected_producer_generations[] = {
    context.expected_producer_generations.state,
    context.expected_producer_generations.conservative_state_gradient,
    context.expected_producer_generations.auxiliary,
    context.expected_producer_generations.trace,
    context.expected_producer_generations.normal,
    context.expected_producer_generations.base_characteristic_speed};
  constexpr int evaluated_field_count =
    static_cast<int>(sizeof(fields) / sizeof(fields[0]));
  static_assert(sizeof(required) / sizeof(required[0]) ==
                  sizeof(fields) / sizeof(fields[0]),
                "required-field inventory must match evaluated fields");
  static_assert(sizeof(expected_producer_generations) /
                    sizeof(expected_producer_generations[0]) ==
                  sizeof(fields) / sizeof(fields[0]),
                "producer-authority inventory must match evaluated fields");
  for (int index = 0; index < evaluated_field_count; ++index)
  {
    if (!required[index])
    {
      continue;
    }
    const ValidationResult validation =
      ValidateRequiredField(*fields[index],
                            batch,
                            expected_producer_generations[index]);
    if (!validation.Ok())
    {
      return validation;
    }
  }

  if (requirements.base_characteristic_speed &&
      batch.base_characteristic_speed.ComponentExtent() != 1)
  {
    return {ValidationCode::InvalidComponentCount,
            "base characteristic speed must have exactly one scalar "
            "component per evaluated point"};
  }

  bool output_components_fit = false;
  const int expected_output_components =
    ExpectedOutputComponents(operation, batch, output_components_fit);
  if (!output_components_fit || expected_output_components <= 0)
  {
    return {ValidationCode::InvalidComponentCount,
            "operation output shape exceeds MFEM Vector limits"};
  }

  const EvaluatedPointOutputBinding& output =
    SelectedOutput(operation, outputs);
  if (output.PointCount() != batch.point_count ||
      output.ComponentExtent() != expected_output_components ||
      output.Dimension() != batch.dimension)
  {
    return {ValidationCode::InvalidBufferSize,
            "operation output extent does not match its evaluated batch"};
  }
  if (output.Site() != batch.site)
  {
    return {ValidationCode::InvalidEvaluationSite,
            "operation output site does not match its evaluated batch"};
  }
  if (output.PointSetIdentity() == 0u ||
      output.PointSetIdentity() != batch.point_set_id)
  {
    return {ValidationCode::InvalidPointSetIdentity,
            "operation output point-set identity does not match its batch"};
  }

  for (int index = 0; index < evaluated_field_count; ++index)
  {
    if (required[index] && fields[index]->Storage() == nullptr)
    {
      return {ValidationCode::MissingBuffer,
              "required evaluated field storage is missing"};
    }
  }
  if (output.Storage() == nullptr)
  {
    return {ValidationCode::MissingBuffer,
            "selected operation output storage is missing"};
  }

  for (int index = 0; index < evaluated_field_count; ++index)
  {
    if (fields[index]->Storage() != nullptr &&
        output.Storage() == fields[index]->Storage())
    {
      return {ValidationCode::InvalidBufferAliasing,
              "writable operation output aliases evaluated input storage"};
    }
  }

  for (int index = 0; index < evaluated_field_count; ++index)
  {
    if (!required[index])
    {
      continue;
    }
    int expected_size = 0;
    if (!ProductFits(fields[index]->ComponentExtent(),
                     fields[index]->PointCount(),
                     expected_size))
    {
      return {ValidationCode::InvalidComponentCount,
              "evaluated input shape exceeds MFEM Vector limits"};
    }
    if (fields[index]->Storage()->Size() != expected_size)
    {
      return {ValidationCode::InvalidBufferSize,
              "evaluated input storage does not match its explicit shape"};
    }
  }

  int expected_output_size = 0;
  if (!ProductFits(expected_output_components,
                   batch.point_count,
                   expected_output_size))
  {
    return {ValidationCode::InvalidComponentCount,
            "operation output shape exceeds MFEM Vector limits"};
  }
  if (output.Storage()->Size() != expected_output_size)
  {
    return {ValidationCode::InvalidBufferSize,
            "operation output storage does not match its explicit shape"};
  }
  return {};
}

} // namespace regularization
} // namespace igrlab

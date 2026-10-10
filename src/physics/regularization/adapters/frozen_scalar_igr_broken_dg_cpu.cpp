#include "igrlab/compat.hpp"
#include "physics/regularization/adapters/frozen_scalar_igr_broken_dg_cpu.hpp"

#include <cmath>

namespace igrlab
{
namespace regularization
{
namespace
{

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

EvaluatedPointFieldMetadata
FieldMetadata(FrozenScalarIGREvaluatedPointStamp stamp,
              int component_extent,
              std::uint64_t producer_generation)
{
  return {stamp.point_count,
          component_extent,
          stamp.dimension,
          stamp.site,
          stamp.point_set_id,
          stamp.rk_stage,
          stamp.state_generation,
          producer_generation};
}

EvaluatedPointBatch BatchHeader(FrozenScalarIGREvaluatedPointStamp stamp)
{
  EvaluatedPointBatch batch;
  batch.point_count = stamp.point_count;
  batch.dimension = stamp.dimension;
  batch.site = stamp.site;
  batch.point_set_id = stamp.point_set_id;
  batch.rk_stage = stamp.rk_stage;
  batch.state_generation = stamp.state_generation;
  return batch;
}

void BindStateAndSigma(EvaluatedPointBatch& batch,
                       const mfem::Vector& evaluated_state,
                       const mfem::Vector& evaluated_sigma,
                       FrozenScalarIGREvaluatedPointStamp stamp)
{
  batch.state = BindEvaluatedPointField(
    evaluated_state,
    FieldMetadata(stamp,
                  FrozenScalarIGREulerComponents(stamp.dimension),
                  stamp.producer_generations.state));
  batch.auxiliary = BindEvaluatedPointField(
    evaluated_sigma,
    FieldMetadata(stamp, 1, stamp.producer_generations.auxiliary));
}

void BindState(EvaluatedPointBatch& batch,
               const mfem::Vector& evaluated_state,
               FrozenScalarIGREvaluatedPointStamp stamp)
{
  batch.state = BindEvaluatedPointField(
    evaluated_state,
    FieldMetadata(stamp,
                  FrozenScalarIGREulerComponents(stamp.dimension),
                  stamp.producer_generations.state));
}

void BindNormal(EvaluatedPointBatch& batch,
                const mfem::Vector& evaluated_normal,
                FrozenScalarIGREvaluatedPointStamp stamp)
{
  batch.normal = BindEvaluatedPointField(
    evaluated_normal,
    FieldMetadata(stamp, stamp.dimension, stamp.producer_generations.normal));
}

RegularizationDevicePolicy
MakePolicy(const FrozenScalarIGRParameters& parameters)
{
  return {parameters.alpha,
          parameters.solver_density_floor,
          parameters.flux_density_floor,
          parameters.sigma_aware_characteristic_speed
            ? static_cast<mfem::real_t>(1.0)
            : static_cast<mfem::real_t>(0.0),
          static_cast<mfem::real_t>(0.0)};
}

/** The accepted generic launcher derives coefficient/source output extent
 * from auxiliary metadata. Supply scalar output shape on a private launch copy
 * without binding, reading, or requiring current sigma storage or authority.
 */
EvaluatedPointBatch WithScalarAuxiliaryOutputShape(
  const EvaluatedPointBatch& batch)
{
  EvaluatedPointBatch shaped = batch;
  shaped.auxiliary = EvaluatedPointFieldBinding::Missing(
    {batch.point_count,
     1,
     batch.dimension,
     batch.site,
     batch.point_set_id,
     batch.rk_stage,
     batch.state_generation,
     0u});
  return shaped;
}

} // namespace

EvaluatedPointBatch
MakeFrozenScalarIGRCoefficientBatch(const mfem::Vector& evaluated_state,
                                    FrozenScalarIGREvaluatedPointStamp stamp)
{
  EvaluatedPointBatch batch = BatchHeader(stamp);
  BindState(batch, evaluated_state, stamp);
  return batch;
}

EvaluatedPointBatch MakeFrozenScalarIGRSourceBatch(
  const mfem::Vector& evaluated_state,
  const mfem::Vector& evaluated_conservative_state_gradient,
  FrozenScalarIGREvaluatedPointStamp stamp)
{
  EvaluatedPointBatch batch = BatchHeader(stamp);
  BindState(batch, evaluated_state, stamp);
  batch.conservative_state_gradient = BindEvaluatedPointField(
    evaluated_conservative_state_gradient,
    FieldMetadata(stamp,
                  FrozenScalarIGREulerComponents(stamp.dimension) *
                    stamp.dimension,
                  stamp.producer_generations.conservative_state_gradient));
  return batch;
}

EvaluatedPointBatch MakeFrozenScalarIGRVolumeCorrectionBatch(
  const mfem::Vector& evaluated_state,
  const mfem::Vector& evaluated_sigma,
  FrozenScalarIGREvaluatedPointStamp stamp)
{
  EvaluatedPointBatch batch = BatchHeader(stamp);
  BindStateAndSigma(batch, evaluated_state, evaluated_sigma, stamp);
  return batch;
}

EvaluatedPointBatch MakeFrozenScalarIGRNormalCorrectionBatch(
  const mfem::Vector& evaluated_state,
  const mfem::Vector& evaluated_sigma,
  const mfem::Vector& evaluated_normal,
  FrozenScalarIGREvaluatedPointStamp stamp)
{
  EvaluatedPointBatch batch =
    MakeFrozenScalarIGRVolumeCorrectionBatch(evaluated_state,
                                             evaluated_sigma,
                                             stamp);
  BindNormal(batch, evaluated_normal, stamp);
  return batch;
}

EvaluatedPointBatch MakeFrozenScalarIGRCharacteristicBoundsBatch(
  const mfem::Vector& evaluated_state,
  const mfem::Vector& evaluated_sigma,
  const mfem::Vector& evaluated_normal,
  const mfem::Vector& evaluated_base_characteristic_speed,
  FrozenScalarIGREvaluatedPointStamp stamp)
{
  EvaluatedPointBatch batch =
    MakeFrozenScalarIGRNormalCorrectionBatch(evaluated_state,
                                             evaluated_sigma,
                                             evaluated_normal,
                                             stamp);
  batch.base_characteristic_speed = BindEvaluatedPointField(
    evaluated_base_characteristic_speed,
    FieldMetadata(stamp,
                  1,
                  stamp.producer_generations.base_characteristic_speed));
  return batch;
}

ValidationResult ValidateFrozenScalarIGRAdapterSelection(
  const FrozenScalarIGRAdapterSelection& selection)
{
  if ((selection.model != FrozenScalarIGRAdapterModelKind::FrozenScalarIGR &&
       selection.model != FrozenScalarIGRAdapterModelKind::Disabled) ||
      !Known(selection.gradient_treatment) ||
      !Known(selection.auxiliary_discretization) ||
      !Known(selection.auxiliary_solution) ||
      !Known(selection.backend.spatial) ||
      !Known(selection.backend.execution) ||
      !Known(selection.compatibility.time) ||
      !Known(selection.compatibility.entropy))
  {
    return {ValidationCode::UnknownAxisValue,
            "frozen scalar IGR adapter selection contains an unknown value"};
  }

  if (selection.model != FrozenScalarIGRAdapterModelKind::FrozenScalarIGR ||
      selection.gradient_treatment !=
        GradientTreatmentKind::ConservativeStateBrokenDG ||
      selection.auxiliary_discretization !=
        AuxiliaryDiscretizationKind::ElementL2 ||
      selection.auxiliary_solution !=
        AuxiliarySolutionKind::GlobalLinearSolve ||
      selection.backend.spatial != SpatialBackendKind::DG ||
      selection.backend.execution != ExecutionTargetKind::MFEMCpu ||
      selection.compatibility.time != TimeTreatmentKind::FrozenRKStage ||
      selection.compatibility.entropy != EntropyTreatmentKind::Neutral)
  {
    return {ValidationCode::UnsupportedCombination,
            "only ElementL2/global-solve/BrokenDG/MFEMCpu/frozen-stage/neutral "
            "frozen scalar IGR is supported"};
  }
  return {};
}

ValidationResult
ValidateFrozenScalarIGRParameters(const FrozenScalarIGRParameters& parameters)
{
  if (!std::isfinite(parameters.alpha) || parameters.alpha < 0.0 ||
      !std::isfinite(parameters.solver_density_floor) ||
      parameters.solver_density_floor <= 0.0 ||
      !std::isfinite(parameters.flux_density_floor) ||
      parameters.flux_density_floor <= 0.0)
  {
    return {ValidationCode::InvalidModelShape,
            "frozen scalar IGR parameters require finite nonnegative alpha "
            "and positive density floors"};
  }
  return {};
}

RegularizationLaunchResult FrozenScalarIGRBrokenDGCPUAdapter::Launch(
  RegularizationOperationKind operation,
  const RegularizationValidationContext& context,
  const EvaluatedPointBatch& batch,
  const RegularizationOperationOutputs& outputs)
{
  if (!IsConfigured())
  {
    return {configuration_validation_, false, host_selection_count_};
  }
  if (!Known(operation))
  {
    return {{ValidationCode::UnknownAxisValue,
             "frozen scalar IGR launch contains an unknown operation"},
            false,
            host_selection_count_};
  }

  const RegularizationStaticLaunchFunction static_launch =
    static_functions_.Select(operation);
  if (static_launch == nullptr)
  {
    return {
      {ValidationCode::UnsupportedCapability,
       "frozen scalar IGR does not provide wall or diagnostics operations"},
      false,
      host_selection_count_};
  }

  const bool scalar_auxiliary_output =
    operation == RegularizationOperationKind::AuxiliaryCoefficient ||
    operation == RegularizationOperationKind::AuxiliarySource;
  const EvaluatedPointBatch launch_batch =
    scalar_auxiliary_output ? WithScalarAuxiliaryOutputShape(batch) : batch;
  RegularizationLaunchResult result =
    static_launch(context, policy_, launch_batch, outputs);
  result.host_selection_count = host_selection_count_;
  if (result.launched)
  {
    ++launch_count_;
  }
  return result;
}

FrozenScalarIGRBrokenDGCPUAdapter ConfigureFrozenScalarIGRBrokenDGCPUAdapter(
  const FrozenScalarIGRAdapterSelection& selection,
  const FrozenScalarIGRParameters& parameters)
{
  const ValidationResult selection_validation =
    ValidateFrozenScalarIGRAdapterSelection(selection);
  if (!selection_validation.Ok())
  {
    return {parameters, {}, {}, selection_validation, 0};
  }

  const ValidationResult parameter_validation =
    ValidateFrozenScalarIGRParameters(parameters);
  if (!parameter_validation.Ok())
  {
    return {parameters, {}, {}, parameter_validation, 0};
  }

  int host_selection_count = 0;
  const RegularizationStaticFunctionTable static_functions =
    detail::BindStaticRegularization<FrozenScalarIGRBrokenDGModelTag,
                                     ElementL2Tag,
                                     GlobalLinearSolveTag,
                                     DGCPUBackendTag,
                                     ConservativeStateBrokenDGTag,
                                     FrozenRKStageTag,
                                     EntropyNeutralTag>(host_selection_count);
  if (!static_functions.HasAny())
  {
    return {parameters,
            {},
            {},
            {ValidationCode::UnsupportedCombination,
             "frozen scalar IGR static operation table did not bind"},
            host_selection_count};
  }
  return {parameters,
          MakePolicy(parameters),
          static_functions,
          {},
          host_selection_count};
}

} // namespace regularization
} // namespace igrlab

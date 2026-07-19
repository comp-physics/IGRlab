#pragma once

#include "physics/regularization/execution.hpp"
#include "physics/regularization/models/frozen_scalar_igr_broken_dg.hpp"

#include <cstdint>

namespace dg_igr
{
namespace regularization
{

enum class FrozenScalarIGRAdapterModelKind : std::uint8_t
{
  Disabled,
  FrozenScalarIGR
};

/** Adapter-local opt-in selection. It is intentionally absent from shared config. */
struct FrozenScalarIGRAdapterSelection
{
  FrozenScalarIGRAdapterModelKind model =
    FrozenScalarIGRAdapterModelKind::Disabled;
  GradientTreatmentKind gradient_treatment =
    GradientTreatmentKind::ConservativeStateBrokenDG;
  AuxiliaryDiscretizationKind auxiliary_discretization =
    AuxiliaryDiscretizationKind::ElementL2;
  AuxiliarySolutionKind auxiliary_solution =
    AuxiliarySolutionKind::GlobalLinearSolve;
  SpatialExecutionBackend backend{SpatialBackendKind::DG,
                                  ExecutionTargetKind::MFEMCpu};
  RegularizationCompatibility compatibility{TimeTreatmentKind::FrozenRKStage,
                                            EntropyTreatmentKind::Neutral};
};

struct FrozenScalarIGREvaluatedPointStamp
{
  int point_count = 0;
  int dimension = 0;
  EvaluationSiteKind site = EvaluationSiteKind::Volume;
  std::uint64_t point_set_id = 0u;
  int rk_stage = -1;
  std::uint64_t state_generation = 0u;
  ExpectedEvaluatedPointProducerGenerations producer_generations{};
};

EvaluatedPointBatch
MakeFrozenScalarIGRCoefficientBatch(const mfem::Vector& evaluated_state,
                                    FrozenScalarIGREvaluatedPointStamp stamp);

EvaluatedPointBatch MakeFrozenScalarIGRSourceBatch(
  const mfem::Vector& evaluated_state,
  const mfem::Vector& evaluated_conservative_state_gradient,
  FrozenScalarIGREvaluatedPointStamp stamp);

EvaluatedPointBatch MakeFrozenScalarIGRVolumeCorrectionBatch(
  const mfem::Vector& evaluated_state,
  const mfem::Vector& evaluated_sigma,
  FrozenScalarIGREvaluatedPointStamp stamp);

EvaluatedPointBatch MakeFrozenScalarIGRNormalCorrectionBatch(
  const mfem::Vector& evaluated_state,
  const mfem::Vector& evaluated_sigma,
  const mfem::Vector& evaluated_normal,
  FrozenScalarIGREvaluatedPointStamp stamp);

EvaluatedPointBatch MakeFrozenScalarIGRCharacteristicBoundsBatch(
  const mfem::Vector& evaluated_state,
  const mfem::Vector& evaluated_sigma,
  const mfem::Vector& evaluated_normal,
  const mfem::Vector& evaluated_base_characteristic_speed,
  FrozenScalarIGREvaluatedPointStamp stamp);

ValidationResult ValidateFrozenScalarIGRAdapterSelection(
  const FrozenScalarIGRAdapterSelection& selection);

ValidationResult
ValidateFrozenScalarIGRParameters(const FrozenScalarIGRParameters& parameters);

class FrozenScalarIGRBrokenDGCPUAdapter
{
public:
  bool IsConfigured() const
  {
    return configuration_validation_.Ok() && static_functions_.HasAny();
  }
  const ValidationResult& ConfigurationValidation() const
  {
    return configuration_validation_;
  }
  const FrozenScalarIGRParameters& Parameters() const { return parameters_; }
  int HostSelectionCount() const { return host_selection_count_; }
  int LaunchCount() const { return launch_count_; }

  RegularizationLaunchResult
  Launch(RegularizationOperationKind operation,
         const RegularizationValidationContext& context,
         const EvaluatedPointBatch& batch,
         const RegularizationOperationOutputs& outputs);

private:
  FrozenScalarIGRBrokenDGCPUAdapter(
    FrozenScalarIGRParameters parameters,
    RegularizationDevicePolicy policy,
    RegularizationStaticFunctionTable static_functions,
    ValidationResult configuration_validation,
    int host_selection_count)
    : parameters_(parameters),
      policy_(policy),
      static_functions_(static_functions),
      configuration_validation_(configuration_validation),
      host_selection_count_(host_selection_count)
  {
  }

  FrozenScalarIGRParameters parameters_{};
  RegularizationDevicePolicy policy_{};
  RegularizationStaticFunctionTable static_functions_{};
  ValidationResult configuration_validation_{};
  int host_selection_count_ = 0;
  int launch_count_ = 0;

  friend FrozenScalarIGRBrokenDGCPUAdapter
  ConfigureFrozenScalarIGRBrokenDGCPUAdapter(
    const FrozenScalarIGRAdapterSelection&,
    const FrozenScalarIGRParameters&);
};

FrozenScalarIGRBrokenDGCPUAdapter ConfigureFrozenScalarIGRBrokenDGCPUAdapter(
  const FrozenScalarIGRAdapterSelection& selection,
  const FrozenScalarIGRParameters& parameters);

} // namespace regularization
} // namespace dg_igr

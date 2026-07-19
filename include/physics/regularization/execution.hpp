#pragma once

#include "physics/regularization/combination.hpp"
#include "physics/regularization/validation.hpp"

#include "mfem.hpp"

#include <type_traits>

namespace dg_igr
{
namespace regularization
{

struct RegularizationLaunchResult
{
  ValidationResult validation{};
  bool launched = false;
  int host_selection_count = 0;
};

using RegularizationStaticLaunchFunction =
  RegularizationLaunchResult (*)(const RegularizationValidationContext&,
                                 const RegularizationDevicePolicy&,
                                 const EvaluatedPointBatch&,
                                 const RegularizationOperationOutputs&);

struct RegularizationStaticFunctionTable
{
  RegularizationStaticLaunchFunction auxiliary_coefficient = nullptr;
  RegularizationStaticLaunchFunction auxiliary_source = nullptr;
  RegularizationStaticLaunchFunction volume_correction = nullptr;
  RegularizationStaticLaunchFunction normal_correction = nullptr;
  RegularizationStaticLaunchFunction wall_law = nullptr;
  RegularizationStaticLaunchFunction characteristic_bounds = nullptr;
  RegularizationStaticLaunchFunction diagnostics = nullptr;

  bool HasAny() const
  {
    return auxiliary_coefficient != nullptr || auxiliary_source != nullptr ||
           volume_correction != nullptr || normal_correction != nullptr ||
           wall_law != nullptr || characteristic_bounds != nullptr ||
           diagnostics != nullptr;
  }

  RegularizationStaticLaunchFunction
  Select(RegularizationOperationKind operation) const
  {
    switch (operation)
    {
    case RegularizationOperationKind::AuxiliaryCoefficient:
      return auxiliary_coefficient;
    case RegularizationOperationKind::AuxiliarySource:
      return auxiliary_source;
    case RegularizationOperationKind::VolumeCorrection:
      return volume_correction;
    case RegularizationOperationKind::NormalCorrection:
      return normal_correction;
    case RegularizationOperationKind::WallLaw:
      return wall_law;
    case RegularizationOperationKind::CharacteristicBounds:
      return characteristic_bounds;
    case RegularizationOperationKind::Diagnostics:
      return diagnostics;
    }
    return nullptr;
  }
};

/** One model/axis configuration with a host-only static operation table. */
class ConfiguredRegularizationLauncher
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
  int HostSelectionCount() const { return host_selection_count_; }
  int LaunchCount() const { return launch_count_; }

  RegularizationLaunchResult
  Launch(RegularizationOperationKind operation,
         const RegularizationValidationContext& context,
         const RegularizationDevicePolicy& policy,
         const EvaluatedPointBatch& batch,
         const RegularizationOperationOutputs& outputs)
  {
    if (!IsConfigured())
    {
      return {configuration_validation_, false, host_selection_count_};
    }

    const bool known_operation =
      operation == RegularizationOperationKind::AuxiliaryCoefficient ||
      operation == RegularizationOperationKind::AuxiliarySource ||
      operation == RegularizationOperationKind::VolumeCorrection ||
      operation == RegularizationOperationKind::NormalCorrection ||
      operation == RegularizationOperationKind::WallLaw ||
      operation == RegularizationOperationKind::CharacteristicBounds ||
      operation == RegularizationOperationKind::Diagnostics;
    if (!known_operation)
    {
      return {{ValidationCode::UnknownAxisValue,
               "launch contains an unknown operation"},
              false,
              host_selection_count_};
    }

    const RegularizationStaticLaunchFunction static_launch =
      static_functions_.Select(operation);
    if (static_launch == nullptr)
    {
      return {{ValidationCode::UnsupportedCapability,
               "selected model does not provide this operation"},
              false,
              host_selection_count_};
    }

    RegularizationLaunchResult result =
      static_launch(context, policy, batch, outputs);
    result.host_selection_count = host_selection_count_;
    if (result.launched)
    {
      ++launch_count_;
    }
    return result;
  }

private:
  ConfiguredRegularizationLauncher(
    ExecutionTargetKind execution_target,
    RegularizationStaticFunctionTable static_functions,
    ValidationResult configuration_validation,
    int host_selection_count)
    : execution_target_(execution_target),
      static_functions_(static_functions),
      configuration_validation_(configuration_validation),
      host_selection_count_(host_selection_count)
  {
  }

  ExecutionTargetKind execution_target_ = ExecutionTargetKind::MFEMCpu;
  RegularizationStaticFunctionTable static_functions_{};
  ValidationResult configuration_validation_{};
  int host_selection_count_ = 0;
  int launch_count_ = 0;

  friend ConfiguredRegularizationLauncher
  ConfigureRegularization(const RegularizationSelection& selection);
};

namespace detail
{

template <typename OperationTag>
const EvaluatedPointOutputBinding&
StaticSelectedOutput(const RegularizationOperationOutputs& outputs)
{
  if constexpr (std::is_same<OperationTag,
                             AuxiliaryCoefficientOperationTag>::value)
  {
    return outputs.auxiliary_coefficient;
  }
  else if constexpr (std::is_same<OperationTag,
                                  AuxiliarySourceOperationTag>::value)
  {
    return outputs.auxiliary_source;
  }
  else if constexpr (std::is_same<OperationTag,
                                  VolumeCorrectionOperationTag>::value)
  {
    return outputs.volume_correction;
  }
  else if constexpr (std::is_same<OperationTag,
                                  NormalCorrectionOperationTag>::value)
  {
    return outputs.normal_correction;
  }
  else if constexpr (std::is_same<OperationTag, WallLawOperationTag>::value)
  {
    return outputs.wall_flux;
  }
  else if constexpr (std::is_same<OperationTag,
                                  CharacteristicBoundsOperationTag>::value)
  {
    return outputs.characteristic_bounds;
  }
  else
  {
    static_assert(std::is_same<OperationTag, DiagnosticsOperationTag>::value,
                  "unknown operation tag");
    return outputs.diagnostics;
  }
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag,
          typename EntropyTag,
          typename OperationTag>
RegularizationLaunchResult
LaunchStaticOperation(const RegularizationValidationContext& context,
                      const RegularizationDevicePolicy& policy,
                      const EvaluatedPointBatch& batch,
                      const RegularizationOperationOutputs& outputs)
{
  using Traits = RegularizationTraits<ModelTag>;
  static_assert(OperationCapabilityEnabled<ModelTag, OperationTag>(),
                "only enabled capabilities may bind a static operation");
  constexpr EvaluatedPointFieldRequirements requirements =
    RegularizationOperationRequirements<ModelTag, OperationTag>::value;

  const EvaluatedPointBatchMetadata metadata = batch.Metadata();
  const ModelPreflightResult model_preflight =
    Traits::ModelPreflight(OperationTag::kind, metadata);
  if (!model_preflight.valid)
  {
    return {{ValidationCode::InvalidModelShape, model_preflight.message},
            false,
            0};
  }

  const ValidationResult validation = ValidateConfiguredRegularizationLaunch(
    BackendTag::use_device ? ExecutionTargetKind::MFEMDevice
                           : ExecutionTargetKind::MFEMCpu,
    OperationTag::kind,
    requirements,
    context,
    batch,
    outputs);
  if (!validation.Ok())
  {
    return {validation, false, 0};
  }

  (void)sizeof(DiscretizationTag);
  (void)sizeof(SolutionTag);
  (void)sizeof(GradientTag);
  (void)sizeof(TimeTag);
  (void)sizeof(EntropyTag);

  constexpr bool use_device = BackendTag::use_device;
  const int point_count = batch.point_count;
  const int dimension = batch.dimension;
  const int state_components = batch.state.ComponentExtent();
  const int auxiliary_components = batch.auxiliary.ComponentExtent();
  const int trace_components = batch.trace.ComponentExtent();
  const int gradient_components =
    batch.conservative_state_gradient.ComponentExtent();
  const int normal_components = batch.normal.ComponentExtent();
  const int base_characteristic_speed_components =
    batch.base_characteristic_speed.ComponentExtent();

  const mfem::real_t* state = nullptr;
  const mfem::real_t* auxiliary = nullptr;
  const mfem::real_t* trace = nullptr;
  const mfem::real_t* conservative_state_gradient = nullptr;
  const mfem::real_t* normal = nullptr;
  const mfem::real_t* base_characteristic_speed = nullptr;
  if constexpr (requirements.state)
  {
    state = batch.state.Read(use_device);
  }
  if constexpr (requirements.auxiliary)
  {
    auxiliary = batch.auxiliary.Read(use_device);
  }
  if constexpr (requirements.trace)
  {
    trace = batch.trace.Read(use_device);
  }
  if constexpr (requirements.conservative_state_gradient)
  {
    conservative_state_gradient =
      batch.conservative_state_gradient.Read(use_device);
  }
  if constexpr (requirements.normal)
  {
    normal = batch.normal.Read(use_device);
  }
  if constexpr (requirements.base_characteristic_speed)
  {
    base_characteristic_speed =
      batch.base_characteristic_speed.Read(use_device);
  }

  const EvaluatedPointOutputBinding& output =
    StaticSelectedOutput<OperationTag>(outputs);
  mfem::real_t* output_data = nullptr;
  if constexpr (std::is_same<OperationTag,
                             VolumeCorrectionOperationTag>::value ||
                std::is_same<OperationTag,
                             NormalCorrectionOperationTag>::value ||
                std::is_same<OperationTag, WallLawOperationTag>::value)
  {
    output_data = output.ReadWrite(use_device);
  }
  else
  {
    output_data = output.Write(use_device);
  }

  const int output_components = output.ComponentExtent();
  const RegularizationDevicePolicy kernel_policy = policy;
  mfem::forall_switch(
    use_device,
    point_count,
    [=] MFEM_HOST_DEVICE(int point_index)
    {
      RegularizationPointInput point{};
      point.state =
        state == nullptr ? nullptr : state + state_components * point_index;
      point.auxiliary = auxiliary == nullptr
                          ? nullptr
                          : auxiliary + auxiliary_components * point_index;
      point.trace =
        trace == nullptr ? nullptr : trace + trace_components * point_index;
      point.conservative_state_gradient =
        conservative_state_gradient == nullptr
          ? nullptr
          : conservative_state_gradient + gradient_components * point_index;
      point.normal =
        normal == nullptr ? nullptr : normal + normal_components * point_index;
      point.base_characteristic_speed =
        base_characteristic_speed == nullptr
          ? nullptr
          : base_characteristic_speed +
              base_characteristic_speed_components * point_index;
      point.state_components = state_components;
      point.auxiliary_components = auxiliary_components;
      point.trace_components = trace_components;
      point.correction_components = state_components;
      point.dimension = dimension;

      mfem::real_t* point_output =
        output_data + output_components * point_index;
      if constexpr (std::is_same<OperationTag,
                                 AuxiliaryCoefficientOperationTag>::value)
      {
        Traits::AuxiliaryCoefficient(kernel_policy, point, point_output);
      }
      else if constexpr (std::is_same<OperationTag,
                                      AuxiliarySourceOperationTag>::value)
      {
        Traits::AuxiliarySource(kernel_policy, point, point_output);
      }
      else if constexpr (std::is_same<OperationTag,
                                      VolumeCorrectionOperationTag>::value)
      {
        Traits::AddVolumeCorrection(kernel_policy, point, point_output);
      }
      else if constexpr (std::is_same<OperationTag,
                                      NormalCorrectionOperationTag>::value)
      {
        Traits::AddNormalCorrection(kernel_policy, point, point_output);
      }
      else if constexpr (std::is_same<OperationTag, WallLawOperationTag>::value)
      {
        Traits::ApplyWallLaw(kernel_policy, point, point_output);
      }
      else if constexpr (std::is_same<OperationTag,
                                      CharacteristicBoundsOperationTag>::value)
      {
        const CharacteristicBoundsResult bounds =
          Traits::CharacteristicBounds(kernel_policy, point);
        point_output[0] = bounds.lower;
        point_output[1] = bounds.upper;
      }
      else
      {
        static_assert(
          std::is_same<OperationTag, DiagnosticsOperationTag>::value,
          "unknown operation tag");
        const DiagnosticResult diagnostics =
          Traits::Diagnostics(kernel_policy, point);
        point_output[0] = diagnostics.source_magnitude;
        point_output[1] = diagnostics.correction_l1;
        point_output[2] = diagnostics.wall_magnitude;
      }
    });

  return {{}, true, 0};
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag,
          typename EntropyTag,
          typename OperationTag>
RegularizationStaticLaunchFunction BindStaticOperation()
{
  if constexpr (OperationCapabilityEnabled<ModelTag, OperationTag>())
  {
    return &LaunchStaticOperation<ModelTag,
                                  DiscretizationTag,
                                  SolutionTag,
                                  BackendTag,
                                  GradientTag,
                                  TimeTag,
                                  EntropyTag,
                                  OperationTag>;
  }
  else
  {
    return nullptr;
  }
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag,
          typename EntropyTag>
RegularizationStaticFunctionTable
BindStaticRegularization(int& host_selection_count)
{
  using Supported = StaticCombinationSupport<ModelTag,
                                             DiscretizationTag,
                                             SolutionTag,
                                             BackendTag,
                                             GradientTag,
                                             TimeTag,
                                             EntropyTag>;
  if constexpr (!Supported::value)
  {
    return {};
  }
  else
  {
    using Contract = RegularizationTraitContract<ModelTag>;
    (void)sizeof(Contract);
    ++host_selection_count;
    return {BindStaticOperation<ModelTag,
                                DiscretizationTag,
                                SolutionTag,
                                BackendTag,
                                GradientTag,
                                TimeTag,
                                EntropyTag,
                                AuxiliaryCoefficientOperationTag>(),
            BindStaticOperation<ModelTag,
                                DiscretizationTag,
                                SolutionTag,
                                BackendTag,
                                GradientTag,
                                TimeTag,
                                EntropyTag,
                                AuxiliarySourceOperationTag>(),
            BindStaticOperation<ModelTag,
                                DiscretizationTag,
                                SolutionTag,
                                BackendTag,
                                GradientTag,
                                TimeTag,
                                EntropyTag,
                                VolumeCorrectionOperationTag>(),
            BindStaticOperation<ModelTag,
                                DiscretizationTag,
                                SolutionTag,
                                BackendTag,
                                GradientTag,
                                TimeTag,
                                EntropyTag,
                                NormalCorrectionOperationTag>(),
            BindStaticOperation<ModelTag,
                                DiscretizationTag,
                                SolutionTag,
                                BackendTag,
                                GradientTag,
                                TimeTag,
                                EntropyTag,
                                WallLawOperationTag>(),
            BindStaticOperation<ModelTag,
                                DiscretizationTag,
                                SolutionTag,
                                BackendTag,
                                GradientTag,
                                TimeTag,
                                EntropyTag,
                                CharacteristicBoundsOperationTag>(),
            BindStaticOperation<ModelTag,
                                DiscretizationTag,
                                SolutionTag,
                                BackendTag,
                                GradientTag,
                                TimeTag,
                                EntropyTag,
                                DiagnosticsOperationTag>()};
  }
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag>
RegularizationStaticFunctionTable DispatchEntropy(EntropyTreatmentKind entropy,
                                                  int& host_selection_count)
{
  switch (entropy)
  {
  case EntropyTreatmentKind::Neutral:
    return BindStaticRegularization<ModelTag,
                                    DiscretizationTag,
                                    SolutionTag,
                                    BackendTag,
                                    GradientTag,
                                    TimeTag,
                                    EntropyNeutralTag>(host_selection_count);
  case EntropyTreatmentKind::EntropyConservative:
    return BindStaticRegularization<ModelTag,
                                    DiscretizationTag,
                                    SolutionTag,
                                    BackendTag,
                                    GradientTag,
                                    TimeTag,
                                    EntropyConservativeTag>(
      host_selection_count);
  case EntropyTreatmentKind::EntropyStable:
    return BindStaticRegularization<ModelTag,
                                    DiscretizationTag,
                                    SolutionTag,
                                    BackendTag,
                                    GradientTag,
                                    TimeTag,
                                    EntropyStableTag>(host_selection_count);
  }
  return {};
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag>
RegularizationStaticFunctionTable
DispatchTime(const RegularizationCompatibility& compatibility,
             int& host_selection_count)
{
  switch (compatibility.time)
  {
  case TimeTreatmentKind::FrozenRKStage:
    return DispatchEntropy<ModelTag,
                           DiscretizationTag,
                           SolutionTag,
                           BackendTag,
                           GradientTag,
                           FrozenRKStageTag>(compatibility.entropy,
                                             host_selection_count);
  case TimeTreatmentKind::LaggedStep:
    return DispatchEntropy<ModelTag,
                           DiscretizationTag,
                           SolutionTag,
                           BackendTag,
                           GradientTag,
                           LaggedStepTag>(compatibility.entropy,
                                          host_selection_count);
  case TimeTreatmentKind::FullyCoupled:
    return DispatchEntropy<ModelTag,
                           DiscretizationTag,
                           SolutionTag,
                           BackendTag,
                           GradientTag,
                           FullyCoupledTag>(compatibility.entropy,
                                            host_selection_count);
  }
  return {};
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename SpatialTag,
          typename GradientTag>
RegularizationStaticFunctionTable
DispatchExecutionTarget(const RegularizationSelection& selection,
                        int& host_selection_count)
{
  switch (selection.axes.backend.execution)
  {
  case ExecutionTargetKind::MFEMCpu:
    return DispatchTime<
      ModelTag,
      DiscretizationTag,
      SolutionTag,
      SpatialExecutionBackendTag<SpatialTag, MFEMCpuTargetTag>,
      GradientTag>(selection.compatibility, host_selection_count);
  case ExecutionTargetKind::MFEMDevice:
    return DispatchTime<
      ModelTag,
      DiscretizationTag,
      SolutionTag,
      SpatialExecutionBackendTag<SpatialTag, MFEMDeviceTargetTag>,
      GradientTag>(selection.compatibility, host_selection_count);
  }
  return {};
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename GradientTag>
RegularizationStaticFunctionTable
DispatchSpatialBackend(const RegularizationSelection& selection,
                       int& host_selection_count)
{
  switch (selection.axes.backend.spatial)
  {
  case SpatialBackendKind::DG:
    return DispatchExecutionTarget<ModelTag,
                                   DiscretizationTag,
                                   SolutionTag,
                                   DGSpatialBackendTag,
                                   GradientTag>(selection,
                                                host_selection_count);
  case SpatialBackendKind::DGSEM:
    return DispatchExecutionTarget<ModelTag,
                                   DiscretizationTag,
                                   SolutionTag,
                                   DGSEMSpatialBackendTag,
                                   GradientTag>(selection,
                                                host_selection_count);
  case SpatialBackendKind::HDG:
    return DispatchExecutionTarget<ModelTag,
                                   DiscretizationTag,
                                   SolutionTag,
                                   HDGSpatialBackendTag,
                                   GradientTag>(selection,
                                                host_selection_count);
  }
  return {};
}

template <typename ModelTag, typename DiscretizationTag, typename GradientTag>
RegularizationStaticFunctionTable
DispatchSolution(const RegularizationSelection& selection,
                 int& host_selection_count)
{
  switch (selection.axes.auxiliary_solution)
  {
  case AuxiliarySolutionKind::PointwiseEvaluation:
    return DispatchSpatialBackend<ModelTag,
                                  DiscretizationTag,
                                  PointwiseEvaluationTag,
                                  GradientTag>(selection, host_selection_count);
  case AuxiliarySolutionKind::LocalElimination:
    return DispatchSpatialBackend<ModelTag,
                                  DiscretizationTag,
                                  LocalEliminationTag,
                                  GradientTag>(selection, host_selection_count);
  case AuxiliarySolutionKind::GlobalLinearSolve:
    return DispatchSpatialBackend<ModelTag,
                                  DiscretizationTag,
                                  GlobalLinearSolveTag,
                                  GradientTag>(selection, host_selection_count);
  }
  return {};
}

template <typename ModelTag, typename GradientTag>
RegularizationStaticFunctionTable
DispatchDiscretization(const RegularizationSelection& selection,
                       int& host_selection_count)
{
  switch (selection.axes.auxiliary_discretization)
  {
  case AuxiliaryDiscretizationKind::ElementL2:
    return DispatchSolution<ModelTag, ElementL2Tag, GradientTag>(
      selection,
      host_selection_count);
  case AuxiliaryDiscretizationKind::TraceL2:
    return DispatchSolution<ModelTag, TraceL2Tag, GradientTag>(
      selection,
      host_selection_count);
  case AuxiliaryDiscretizationKind::HybridizedTrace:
    return DispatchSolution<ModelTag, HybridizedTraceTag, GradientTag>(
      selection,
      host_selection_count);
  }
  return {};
}

template <typename ModelTag>
RegularizationStaticFunctionTable
DispatchGradient(const RegularizationSelection& selection,
                 int& host_selection_count)
{
  switch (selection.axes.gradient_treatment)
  {
  case GradientTreatmentKind::None:
    return DispatchDiscretization<ModelTag, NoneGradientTreatmentTag>(
      selection,
      host_selection_count);
  case GradientTreatmentKind::ConservativeStateBrokenDG:
    return DispatchDiscretization<ModelTag, ConservativeStateBrokenDGTag>(
      selection,
      host_selection_count);
  }
  return {};
}

inline RegularizationStaticFunctionTable
DispatchModel(const RegularizationSelection& selection,
              int& host_selection_count)
{
  switch (selection.axes.continuum_model)
  {
  case ContinuumModelKind::ZeroFixture:
    return DispatchGradient<ZeroFixtureModelTag>(selection,
                                                 host_selection_count);
  case ContinuumModelKind::ConstantFixture:
    return DispatchGradient<ConstantFixtureModelTag>(selection,
                                                     host_selection_count);
  case ContinuumModelKind::BrokenDGContractProbe:
    return DispatchGradient<BrokenDGContractProbeModelTag>(
      selection,
      host_selection_count);
  }
  return {};
}

} // namespace detail

inline ConfiguredRegularizationLauncher
ConfigureRegularization(const RegularizationSelection& selection)
{
  const ValidationResult validation =
    ValidateRegularizationSelection(selection);
  if (!validation.Ok())
  {
    return {selection.axes.backend.execution, {}, validation, 0};
  }

  int host_selection_count = 0;
  const RegularizationStaticFunctionTable static_functions =
    detail::DispatchModel(selection, host_selection_count);
  if (!static_functions.HasAny())
  {
    return {selection.axes.backend.execution,
            {},
            {ValidationCode::UnsupportedCombination,
             "validated selection did not bind a static function table"},
            host_selection_count};
  }
  return {selection.axes.backend.execution,
          static_functions,
          {},
          host_selection_count};
}

inline RegularizationLaunchResult
ExecuteRegularization(const RegularizationSelection& selection,
                      RegularizationOperationKind operation,
                      const RegularizationValidationContext& context,
                      const RegularizationDevicePolicy& policy,
                      const EvaluatedPointBatch& batch,
                      const RegularizationOperationOutputs& outputs)
{
  ConfiguredRegularizationLauncher launcher =
    ConfigureRegularization(selection);
  return launcher.Launch(operation, context, policy, batch, outputs);
}

} // namespace regularization
} // namespace dg_igr

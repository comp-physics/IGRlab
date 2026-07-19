#pragma once

#include "physics/regularization/combination.hpp"
#include "physics/regularization/validation.hpp"

#include "mfem.hpp"

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

using RegularizationStaticLaunchFunction = RegularizationLaunchResult (*)(
  const RegularizationDevicePolicy&,
  const RegularizationLaunchBuffers&);

/** Host-configured, reusable entry point for one statically selected contract.
 *
 * Configuration traverses the runtime axes once and binds a specialized host
 * function pointer. Each Launch call validates its current context and buffers,
 * then invokes that function without revisiting model or axis selection.
 */
class ConfiguredRegularizationLauncher
{
public:
  bool IsConfigured() const
  {
    return configuration_validation_.Ok() && static_launch_ != nullptr;
  }
  const ValidationResult& ConfigurationValidation() const
  {
    return configuration_validation_;
  }
  int HostSelectionCount() const { return host_selection_count_; }
  int LaunchCount() const { return launch_count_; }

  RegularizationLaunchResult Launch(
    const RegularizationValidationContext& context,
    const RegularizationDevicePolicy& policy,
    const RegularizationLaunchBuffers& buffers)
  {
    if (!IsConfigured())
    {
      return {configuration_validation_, false, host_selection_count_};
    }
    const ValidationResult validation =
      ValidateConfiguredRegularizationLaunch(execution_target_, context, buffers);
    if (!validation.Ok())
    {
      return {validation, false, host_selection_count_};
    }
    RegularizationLaunchResult result = static_launch_(policy, buffers);
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
    RegularizationStaticLaunchFunction static_launch,
    ValidationResult configuration_validation,
    int host_selection_count)
    : execution_target_(execution_target), static_launch_(static_launch),
      configuration_validation_(configuration_validation),
      host_selection_count_(host_selection_count)
  {
  }

  ExecutionTargetKind execution_target_ = ExecutionTargetKind::MFEMCpu;
  RegularizationStaticLaunchFunction static_launch_ = nullptr;
  ValidationResult configuration_validation_{};
  int host_selection_count_ = 0;
  int launch_count_ = 0;

  friend ConfiguredRegularizationLauncher ConfigureRegularization(
    const RegularizationSelection& selection);
};

namespace detail
{

inline RegularizationLaunchResult DispatchFailure(const char* message)
{
  RegularizationLaunchResult result;
  result.validation = {ValidationCode::UnsupportedCombination, message};
  return result;
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename TimeTag,
          typename EntropyTag>
RegularizationLaunchResult LaunchStaticRegularization(
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  using Supported = StaticCombinationSupport<ModelTag,
                                              DiscretizationTag,
                                              SolutionTag,
                                              BackendTag,
                                              TimeTag,
                                              EntropyTag>;
  if constexpr (!Supported::value)
  {
    return DispatchFailure(
      "validated selection did not map to a supported static combination");
  }
  else
  {
    using Traits = RegularizationTraits<ModelTag>;
    using Contract = RegularizationTraitContract<ModelTag>;
    (void)sizeof(Contract);
    (void)sizeof(DiscretizationTag);
    (void)sizeof(SolutionTag);
    (void)sizeof(typename BackendTag::spatial_backend_tag);
    (void)sizeof(typename BackendTag::execution_target_tag);
    (void)sizeof(TimeTag);
    (void)sizeof(EntropyTag);

    constexpr bool use_device = BackendTag::use_device;
    const int point_count = buffers.point_count;
    const int dimension = buffers.dimension;
    const int state_components = buffers.state_components;
    const int auxiliary_components = buffers.auxiliary.components;
    const int trace_components = buffers.trace.components;
    const int correction_components =
      buffers.volume_correction.components;

    const mfem::real_t* state = buffers.state->Read(use_device);
    const mfem::real_t* auxiliary = buffers.auxiliary.Read(use_device);
    const mfem::real_t* trace = buffers.trace.Read(use_device);
    const mfem::real_t* state_gradient =
      buffers.state_gradient->Read(use_device);
    const mfem::real_t* normal_data = buffers.normals->Read(use_device);

    mfem::real_t* coefficient = buffers.coefficient->Write(use_device);
    mfem::real_t* source = buffers.source->Write(use_device);
    mfem::real_t* volume =
      buffers.volume_correction.storage->ReadWrite(use_device);
    mfem::real_t* normal =
      buffers.normal_correction.storage->ReadWrite(use_device);
    mfem::real_t* wall = buffers.wall_flux.storage->ReadWrite(use_device);
    auto bounds = mfem::Reshape(
      buffers.characteristic_bounds->Write(use_device), 2, point_count);
    auto diagnostics =
      mfem::Reshape(buffers.diagnostics->Write(use_device), 3, point_count);

    const RegularizationDevicePolicy kernel_policy = policy;
    mfem::forall_switch(
      use_device,
      point_count,
      [=] MFEM_HOST_DEVICE(int point_index)
      {
        RegularizationPointInput point{};
        point.state = state + state_components * point_index;
        point.auxiliary =
          auxiliary + auxiliary_components * point_index;
        point.trace = trace + trace_components * point_index;
        point.state_gradient =
          state_gradient + state_components * dimension * point_index;
        point.normal = normal_data + dimension * point_index;
        point.state_components = state_components;
        point.auxiliary_components = auxiliary_components;
        point.trace_components = trace_components;
        point.correction_components = correction_components;
        point.dimension = dimension;

        mfem::real_t* point_coefficient =
          coefficient + auxiliary_components * point_index;
        mfem::real_t* point_source =
          source + auxiliary_components * point_index;
        mfem::real_t* point_volume =
          volume + correction_components * dimension * point_index;
        mfem::real_t* point_normal =
          normal + correction_components * point_index;
        mfem::real_t* point_wall = wall + correction_components * point_index;

        Traits::CoefficientsAndSource(
          kernel_policy, point, point_coefficient, point_source);
        Traits::AddVolumeCorrection(kernel_policy, point, point_volume);
        Traits::AddNormalCorrection(kernel_policy, point, point_normal);
        Traits::ApplyWallLaw(kernel_policy, point, point_wall);

        const CharacteristicBoundsResult point_bounds =
          Traits::CharacteristicBounds(kernel_policy, point);
        bounds(0, point_index) = point_bounds.lower;
        bounds(1, point_index) = point_bounds.upper;

        const DiagnosticResult point_diagnostics =
          Traits::Diagnostics(kernel_policy, point);
        diagnostics(0, point_index) = point_diagnostics.source_magnitude;
        diagnostics(1, point_index) = point_diagnostics.correction_l1;
        diagnostics(2, point_index) = point_diagnostics.wall_magnitude;
      });

    RegularizationLaunchResult result;
    result.launched = true;
    return result;
  }
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename TimeTag,
          typename EntropyTag>
RegularizationStaticLaunchFunction BindStaticRegularization(
  int& host_selection_count)
{
  using Supported = StaticCombinationSupport<ModelTag,
                                              DiscretizationTag,
                                              SolutionTag,
                                              BackendTag,
                                              TimeTag,
                                              EntropyTag>;
  if constexpr (!Supported::value)
  {
    return nullptr;
  }
  else
  {
    ++host_selection_count;
    return &LaunchStaticRegularization<ModelTag,
                                       DiscretizationTag,
                                       SolutionTag,
                                       BackendTag,
                                       TimeTag,
                                       EntropyTag>;
  }
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename TimeTag>
RegularizationStaticLaunchFunction DispatchEntropy(
  EntropyTreatmentKind entropy,
  int& host_selection_count)
{
  // Host-only dispatch: no switch reaches the point kernel.
  switch (entropy)
  {
    case EntropyTreatmentKind::Neutral:
      return BindStaticRegularization<ModelTag,
                                      DiscretizationTag,
                                      SolutionTag,
                                      BackendTag,
                                      TimeTag,
                                      EntropyNeutralTag>(host_selection_count);
    case EntropyTreatmentKind::EntropyConservative:
      return BindStaticRegularization<ModelTag,
                                      DiscretizationTag,
                                      SolutionTag,
                                      BackendTag,
                                      TimeTag,
                                      EntropyConservativeTag>(
        host_selection_count);
    case EntropyTreatmentKind::EntropyStable:
      return BindStaticRegularization<ModelTag,
                                      DiscretizationTag,
                                      SolutionTag,
                                      BackendTag,
                                      TimeTag,
                                      EntropyStableTag>(host_selection_count);
  }
  return nullptr;
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag>
RegularizationStaticLaunchFunction DispatchTime(
  const RegularizationCompatibility& compatibility,
  int& host_selection_count)
{
  switch (compatibility.time)
  {
    case TimeTreatmentKind::FrozenRKStage:
      return DispatchEntropy<ModelTag,
                             DiscretizationTag,
                             SolutionTag,
                             BackendTag,
                             FrozenRKStageTag>(compatibility.entropy,
                                               host_selection_count);
    case TimeTreatmentKind::LaggedStep:
      return DispatchEntropy<ModelTag,
                             DiscretizationTag,
                             SolutionTag,
                             BackendTag,
                             LaggedStepTag>(compatibility.entropy,
                                            host_selection_count);
    case TimeTreatmentKind::FullyCoupled:
      return DispatchEntropy<ModelTag,
                             DiscretizationTag,
                             SolutionTag,
                             BackendTag,
                             FullyCoupledTag>(compatibility.entropy,
                                              host_selection_count);
  }
  return nullptr;
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename SpatialTag>
RegularizationStaticLaunchFunction DispatchExecutionTarget(
  const RegularizationSelection& selection,
  int& host_selection_count)
{
  switch (selection.axes.backend.execution)
  {
    case ExecutionTargetKind::MFEMCpu:
      return DispatchTime<
        ModelTag,
        DiscretizationTag,
        SolutionTag,
        SpatialExecutionBackendTag<SpatialTag, MFEMCpuTargetTag>>(
        selection.compatibility, host_selection_count);
    case ExecutionTargetKind::MFEMDevice:
      return DispatchTime<
        ModelTag,
        DiscretizationTag,
        SolutionTag,
        SpatialExecutionBackendTag<SpatialTag, MFEMDeviceTargetTag>>(
        selection.compatibility, host_selection_count);
  }
  return nullptr;
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag>
RegularizationStaticLaunchFunction DispatchSpatialBackend(
  const RegularizationSelection& selection,
  int& host_selection_count)
{
  switch (selection.axes.backend.spatial)
  {
    case SpatialBackendKind::DG:
      return DispatchExecutionTarget<ModelTag,
                                     DiscretizationTag,
                                     SolutionTag,
                                     DGSpatialBackendTag>(selection,
                                                          host_selection_count);
    case SpatialBackendKind::DGSEM:
      return DispatchExecutionTarget<ModelTag,
                                     DiscretizationTag,
                                     SolutionTag,
                                     DGSEMSpatialBackendTag>(selection,
                                                             host_selection_count);
    case SpatialBackendKind::HDG:
      return DispatchExecutionTarget<ModelTag,
                                     DiscretizationTag,
                                     SolutionTag,
                                     HDGSpatialBackendTag>(selection,
                                                           host_selection_count);
  }
  return nullptr;
}

template <typename ModelTag, typename DiscretizationTag>
RegularizationStaticLaunchFunction DispatchSolution(
  const RegularizationSelection& selection,
  int& host_selection_count)
{
  switch (selection.axes.auxiliary_solution)
  {
    case AuxiliarySolutionKind::PointwiseEvaluation:
      return DispatchSpatialBackend<ModelTag,
                                    DiscretizationTag,
                                    PointwiseEvaluationTag>(selection,
                                                            host_selection_count);
    case AuxiliarySolutionKind::LocalElimination:
      return DispatchSpatialBackend<ModelTag,
                                    DiscretizationTag,
                                    LocalEliminationTag>(selection,
                                                         host_selection_count);
    case AuxiliarySolutionKind::GlobalLinearSolve:
      return DispatchSpatialBackend<ModelTag,
                                    DiscretizationTag,
                                    GlobalLinearSolveTag>(selection,
                                                          host_selection_count);
  }
  return nullptr;
}

template <typename ModelTag>
RegularizationStaticLaunchFunction DispatchDiscretization(
  const RegularizationSelection& selection,
  int& host_selection_count)
{
  switch (selection.axes.auxiliary_discretization)
  {
    case AuxiliaryDiscretizationKind::ElementL2:
      return DispatchSolution<ModelTag, ElementL2Tag>(selection,
                                                       host_selection_count);
    case AuxiliaryDiscretizationKind::TraceL2:
      return DispatchSolution<ModelTag, TraceL2Tag>(selection,
                                                     host_selection_count);
    case AuxiliaryDiscretizationKind::HybridizedTrace:
      return DispatchSolution<ModelTag, HybridizedTraceTag>(selection,
                                                             host_selection_count);
  }
  return nullptr;
}

inline RegularizationStaticLaunchFunction DispatchModel(
  const RegularizationSelection& selection,
  int& host_selection_count)
{
  switch (selection.axes.continuum_model)
  {
    case ContinuumModelKind::ZeroFixture:
      return DispatchDiscretization<ZeroFixtureModelTag>(selection,
                                                          host_selection_count);
    case ContinuumModelKind::ConstantFixture:
      return DispatchDiscretization<ConstantFixtureModelTag>(selection,
                                                              host_selection_count);
  }
  return nullptr;
}

} // namespace detail

/** Validate and freeze every host selection axis into one reusable launcher. */
inline ConfiguredRegularizationLauncher ConfigureRegularization(
  const RegularizationSelection& selection)
{
  const ValidationResult validation =
    ValidateRegularizationSelection(selection);
  if (!validation.Ok())
  {
    return {selection.axes.backend.execution,
            nullptr,
            validation,
            0};
  }

  int host_selection_count = 0;
  const RegularizationStaticLaunchFunction static_launch =
    detail::DispatchModel(selection, host_selection_count);
  if (static_launch == nullptr)
  {
    return {selection.axes.backend.execution,
            nullptr,
            {ValidationCode::UnsupportedCombination,
             "validated selection did not bind a static launch function"},
            host_selection_count};
  }
  return {selection.axes.backend.execution,
          static_launch,
          {},
          host_selection_count};
}

/** One-shot convenience wrapper. Repeated launches should configure once and
 * reuse ConfiguredRegularizationLauncher::Launch directly. */
inline RegularizationLaunchResult ExecuteRegularization(
  const RegularizationSelection& selection,
  const RegularizationValidationContext& context,
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  ConfiguredRegularizationLauncher launcher =
    ConfigureRegularization(selection);
  return launcher.Launch(context, policy, buffers);
}

} // namespace regularization
} // namespace dg_igr

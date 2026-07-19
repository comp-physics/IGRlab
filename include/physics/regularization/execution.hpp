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

namespace detail
{

inline RegularizationLaunchResult DispatchFailure(const char* message)
{
  return {{ValidationCode::UnsupportedCombination, message}, false, 1};
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

    return {{}, true, 1};
  }
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename TimeTag>
RegularizationLaunchResult DispatchEntropy(
  EntropyTreatmentKind entropy,
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  // Host-only dispatch: no switch reaches the point kernel.
  switch (entropy)
  {
    case EntropyTreatmentKind::Neutral:
      return LaunchStaticRegularization<ModelTag,
                                        DiscretizationTag,
                                        SolutionTag,
                                        BackendTag,
                                        TimeTag,
                                        EntropyNeutralTag>(policy, buffers);
    case EntropyTreatmentKind::EntropyConservative:
      return LaunchStaticRegularization<ModelTag,
                                        DiscretizationTag,
                                        SolutionTag,
                                        BackendTag,
                                        TimeTag,
                                        EntropyConservativeTag>(policy, buffers);
    case EntropyTreatmentKind::EntropyStable:
      return LaunchStaticRegularization<ModelTag,
                                        DiscretizationTag,
                                        SolutionTag,
                                        BackendTag,
                                        TimeTag,
                                        EntropyStableTag>(policy, buffers);
  }
  return DispatchFailure("unknown entropy treatment during host dispatch");
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag>
RegularizationLaunchResult DispatchTime(
  const RegularizationCompatibility& compatibility,
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  switch (compatibility.time)
  {
    case TimeTreatmentKind::FrozenRKStage:
      return DispatchEntropy<ModelTag,
                             DiscretizationTag,
                             SolutionTag,
                             BackendTag,
                             FrozenRKStageTag>(compatibility.entropy,
                                               policy,
                                               buffers);
    case TimeTreatmentKind::LaggedStep:
      return DispatchEntropy<ModelTag,
                             DiscretizationTag,
                             SolutionTag,
                             BackendTag,
                             LaggedStepTag>(compatibility.entropy,
                                            policy,
                                            buffers);
    case TimeTreatmentKind::FullyCoupled:
      return DispatchEntropy<ModelTag,
                             DiscretizationTag,
                             SolutionTag,
                             BackendTag,
                             FullyCoupledTag>(compatibility.entropy,
                                              policy,
                                              buffers);
  }
  return DispatchFailure("unknown time treatment during host dispatch");
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename SpatialTag>
RegularizationLaunchResult DispatchExecutionTarget(
  const RegularizationSelection& selection,
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  switch (selection.axes.backend.execution)
  {
    case ExecutionTargetKind::MFEMCpu:
      return DispatchTime<
        ModelTag,
        DiscretizationTag,
        SolutionTag,
        SpatialExecutionBackendTag<SpatialTag, MFEMCpuTargetTag>>(
        selection.compatibility, policy, buffers);
    case ExecutionTargetKind::MFEMDevice:
      return DispatchTime<
        ModelTag,
        DiscretizationTag,
        SolutionTag,
        SpatialExecutionBackendTag<SpatialTag, MFEMDeviceTargetTag>>(
        selection.compatibility, policy, buffers);
  }
  return DispatchFailure("unknown execution target during host dispatch");
}

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag>
RegularizationLaunchResult DispatchSpatialBackend(
  const RegularizationSelection& selection,
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  switch (selection.axes.backend.spatial)
  {
    case SpatialBackendKind::DG:
      return DispatchExecutionTarget<ModelTag,
                                     DiscretizationTag,
                                     SolutionTag,
                                     DGSpatialBackendTag>(selection,
                                                          policy,
                                                          buffers);
    case SpatialBackendKind::DGSEM:
      return DispatchExecutionTarget<ModelTag,
                                     DiscretizationTag,
                                     SolutionTag,
                                     DGSEMSpatialBackendTag>(selection,
                                                             policy,
                                                             buffers);
    case SpatialBackendKind::HDG:
      return DispatchExecutionTarget<ModelTag,
                                     DiscretizationTag,
                                     SolutionTag,
                                     HDGSpatialBackendTag>(selection,
                                                           policy,
                                                           buffers);
  }
  return DispatchFailure("unknown spatial backend during host dispatch");
}

template <typename ModelTag, typename DiscretizationTag>
RegularizationLaunchResult DispatchSolution(
  const RegularizationSelection& selection,
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  switch (selection.axes.auxiliary_solution)
  {
    case AuxiliarySolutionKind::PointwiseEvaluation:
      return DispatchSpatialBackend<ModelTag,
                                    DiscretizationTag,
                                    PointwiseEvaluationTag>(selection,
                                                            policy,
                                                            buffers);
    case AuxiliarySolutionKind::LocalElimination:
      return DispatchSpatialBackend<ModelTag,
                                    DiscretizationTag,
                                    LocalEliminationTag>(selection,
                                                         policy,
                                                         buffers);
    case AuxiliarySolutionKind::GlobalLinearSolve:
      return DispatchSpatialBackend<ModelTag,
                                    DiscretizationTag,
                                    GlobalLinearSolveTag>(selection,
                                                          policy,
                                                          buffers);
  }
  return DispatchFailure("unknown auxiliary solve during host dispatch");
}

template <typename ModelTag>
RegularizationLaunchResult DispatchDiscretization(
  const RegularizationSelection& selection,
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  switch (selection.axes.auxiliary_discretization)
  {
    case AuxiliaryDiscretizationKind::ElementL2:
      return DispatchSolution<ModelTag, ElementL2Tag>(selection,
                                                       policy,
                                                       buffers);
    case AuxiliaryDiscretizationKind::TraceL2:
      return DispatchSolution<ModelTag, TraceL2Tag>(selection, policy, buffers);
    case AuxiliaryDiscretizationKind::HybridizedTrace:
      return DispatchSolution<ModelTag, HybridizedTraceTag>(selection,
                                                             policy,
                                                             buffers);
  }
  return DispatchFailure("unknown auxiliary discretization during host dispatch");
}

inline RegularizationLaunchResult DispatchModel(
  const RegularizationSelection& selection,
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  switch (selection.axes.continuum_model)
  {
    case ContinuumModelKind::ZeroFixture:
      return DispatchDiscretization<ZeroFixtureModelTag>(selection,
                                                          policy,
                                                          buffers);
    case ContinuumModelKind::ConstantFixture:
      return DispatchDiscretization<ConstantFixtureModelTag>(selection,
                                                              policy,
                                                              buffers);
  }
  return DispatchFailure("unknown continuum model during host dispatch");
}

} // namespace detail

/** Validate once on the host, select each independent axis once, then execute
 * a fully templated model kernel. Validation failure returns without touching
 * any MFEM data pointer or launching any point work. */
inline RegularizationLaunchResult ExecuteRegularization(
  const RegularizationSelection& selection,
  const RegularizationValidationContext& context,
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  const ValidationResult validation =
    ValidateRegularizationLaunch(selection, context, buffers);
  if (!validation.Ok())
  {
    return {validation, false, 0};
  }
  return detail::DispatchModel(selection, policy, buffers);
}

} // namespace regularization
} // namespace dg_igr

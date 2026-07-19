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
    (void)sizeof(TimeTag);
    (void)sizeof(EntropyTag);

    constexpr bool use_device = BackendTag::use_device;
    const int point_count = buffers.point_count;
    const int dimension = buffers.dimension;

    const auto state =
      mfem::Reshape(buffers.state->Read(use_device), point_count);
    const auto auxiliary =
      mfem::Reshape(buffers.auxiliary->Read(use_device), point_count);
    const auto trace =
      mfem::Reshape(buffers.trace->Read(use_device), point_count);
    const auto gradient = mfem::Reshape(
      buffers.gradient->Read(use_device), dimension, point_count);
    const auto normals = mfem::Reshape(
      buffers.normals->Read(use_device), dimension, point_count);

    auto coefficient =
      mfem::Reshape(buffers.coefficient->Write(use_device), point_count);
    auto source = mfem::Reshape(buffers.source->Write(use_device), point_count);
    auto volume = mfem::Reshape(
      buffers.volume_correction->ReadWrite(use_device), dimension, point_count);
    auto normal = mfem::Reshape(
      buffers.normal_correction->ReadWrite(use_device), point_count);
    auto wall =
      mfem::Reshape(buffers.wall_flux->ReadWrite(use_device), point_count);
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
        point.state = state(point_index);
        point.auxiliary = auxiliary(point_index);
        point.trace = trace(point_index);
        point.dimension = dimension;
        for (int d = 0; d < dimension; ++d)
        {
          point.gradient[d] = gradient(d, point_index);
          point.normal[d] = normals(d, point_index);
        }

        const CoefficientSourceResult coefficient_source =
          Traits::CoefficientsAndSource(kernel_policy, point);
        coefficient(point_index) = coefficient_source.coefficient;
        source(point_index) = coefficient_source.source;

        mfem::real_t point_volume[3] = {0.0, 0.0, 0.0};
        for (int d = 0; d < dimension; ++d)
        {
          point_volume[d] = volume(d, point_index);
        }
        Traits::AddVolumeCorrection(kernel_policy, point, point_volume);
        for (int d = 0; d < dimension; ++d)
        {
          volume(d, point_index) = point_volume[d];
        }
        normal(point_index) += Traits::NormalCorrection(kernel_policy, point);
        wall(point_index) += Traits::WallLaw(kernel_policy, point);

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
          typename SolutionTag>
RegularizationLaunchResult DispatchBackend(
  const RegularizationSelection& selection,
  const RegularizationDevicePolicy& policy,
  const RegularizationLaunchBuffers& buffers)
{
  switch (selection.axes.execution_backend)
  {
    case ExecutionBackendKind::MFEMCpu:
      return DispatchTime<ModelTag,
                          DiscretizationTag,
                          SolutionTag,
                          MFEMCpuBackendTag>(selection.compatibility,
                                             policy,
                                             buffers);
    case ExecutionBackendKind::MFEMDevice:
      return DispatchTime<ModelTag,
                          DiscretizationTag,
                          SolutionTag,
                          MFEMDeviceBackendTag>(selection.compatibility,
                                                policy,
                                                buffers);
  }
  return DispatchFailure("unknown execution backend during host dispatch");
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
      return DispatchBackend<ModelTag,
                             DiscretizationTag,
                             PointwiseEvaluationTag>(selection,
                                                     policy,
                                                     buffers);
    case AuxiliarySolutionKind::LocalElimination:
      return DispatchBackend<ModelTag,
                             DiscretizationTag,
                             LocalEliminationTag>(selection, policy, buffers);
    case AuxiliarySolutionKind::GlobalLinearSolve:
      return DispatchBackend<ModelTag,
                             DiscretizationTag,
                             GlobalLinearSolveTag>(selection, policy, buffers);
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

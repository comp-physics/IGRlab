#pragma once

#include "physics/regularization/axes.hpp"
#include "physics/regularization/traits.hpp"

#include <type_traits>

namespace dg_igr
{
namespace regularization
{
namespace detail
{

template <typename BackendTag, typename = void> struct BackendTagMetadata
{
  static constexpr bool valid = false;
  static constexpr bool is_dg = false;
  static constexpr bool is_dgsem = false;
  static constexpr bool is_hdg = false;
};

template <typename BackendTag>
struct BackendTagMetadata<
  BackendTag,
  std::void_t<typename BackendTag::spatial_backend_tag,
              typename BackendTag::execution_target_tag>>
{
  using SpatialTag = typename BackendTag::spatial_backend_tag;
  using ExecutionTag = typename BackendTag::execution_target_tag;
  static constexpr bool valid_execution =
    std::is_same<ExecutionTag, MFEMCpuTargetTag>::value ||
    std::is_same<ExecutionTag, MFEMDeviceTargetTag>::value;
  static constexpr bool is_dg =
    std::is_same<SpatialTag, DGSpatialBackendTag>::value;
  static constexpr bool is_dgsem =
    std::is_same<SpatialTag, DGSEMSpatialBackendTag>::value;
  static constexpr bool is_hdg =
    std::is_same<SpatialTag, HDGSpatialBackendTag>::value;
  static constexpr bool valid =
    valid_execution && (is_dg || is_dgsem || is_hdg);
};

} // namespace detail

/** Model-owned customization point for the complete combination matrix.
 *
 * Future models specialize this template instead of extending a central
 * model-name Boolean.
 */
template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag,
          typename EntropyTag>
struct ModelCombinationSupport : std::false_type
{
};

template <typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag,
          typename EntropyTag>
struct ModelCombinationSupport<ZeroFixtureModelTag,
                               DiscretizationTag,
                               SolutionTag,
                               BackendTag,
                               GradientTag,
                               TimeTag,
                               EntropyTag>
  : std::integral_constant<
      bool,
      detail::BackendTagMetadata<BackendTag>::valid &&
        std::is_same<GradientTag, NoneGradientTreatmentTag>::value &&
        std::is_same<TimeTag, FrozenRKStageTag>::value &&
        std::is_same<EntropyTag, EntropyNeutralTag>::value &&
        std::is_same<DiscretizationTag, ElementL2Tag>::value &&
        std::is_same<SolutionTag, PointwiseEvaluationTag>::value>
{
};

template <typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag,
          typename EntropyTag>
struct ModelCombinationSupport<ConstantFixtureModelTag,
                               DiscretizationTag,
                               SolutionTag,
                               BackendTag,
                               GradientTag,
                               TimeTag,
                               EntropyTag>
  : std::integral_constant<
      bool,
      detail::BackendTagMetadata<BackendTag>::valid &&
        std::is_same<GradientTag, NoneGradientTreatmentTag>::value &&
        std::is_same<TimeTag, FrozenRKStageTag>::value &&
        std::is_same<EntropyTag, EntropyNeutralTag>::value &&
        ((std::is_same<DiscretizationTag, ElementL2Tag>::value &&
          std::is_same<SolutionTag, PointwiseEvaluationTag>::value) ||
         (std::is_same<DiscretizationTag, TraceL2Tag>::value &&
          std::is_same<SolutionTag, LocalEliminationTag>::value &&
          (detail::BackendTagMetadata<BackendTag>::is_dg ||
           detail::BackendTagMetadata<BackendTag>::is_dgsem)) ||
         (std::is_same<DiscretizationTag, HybridizedTraceTag>::value &&
          std::is_same<SolutionTag, GlobalLinearSolveTag>::value &&
          detail::BackendTagMetadata<BackendTag>::is_hdg))>
{
};

template <typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag,
          typename EntropyTag>
struct ModelCombinationSupport<BrokenDGContractProbeModelTag,
                               DiscretizationTag,
                               SolutionTag,
                               BackendTag,
                               GradientTag,
                               TimeTag,
                               EntropyTag>
  : std::integral_constant<
      bool,
      std::is_same<DiscretizationTag, ElementL2Tag>::value &&
        std::is_same<SolutionTag, GlobalLinearSolveTag>::value &&
        std::is_same<BackendTag, DGCPUBackendTag>::value &&
        std::is_same<GradientTag, ConservativeStateBrokenDGTag>::value &&
        std::is_same<TimeTag, FrozenRKStageTag>::value &&
        std::is_same<EntropyTag, EntropyNeutralTag>::value>
{
};

template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag,
          typename EntropyTag>
struct StaticCombinationSupport
  : std::integral_constant<bool,
                           HasRegularizationTraits<ModelTag>::value &&
                             ModelCombinationSupport<ModelTag,
                                                     DiscretizationTag,
                                                     SolutionTag,
                                                     BackendTag,
                                                     GradientTag,
                                                     TimeTag,
                                                     EntropyTag>::value>
{
};

using SupportedCpuFixtureCombination =
  StaticCombinationSupport<ZeroFixtureModelTag,
                           ElementL2Tag,
                           PointwiseEvaluationTag,
                           DGCPUBackendTag,
                           NoneGradientTreatmentTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

using SupportedDeviceFixtureCombination =
  StaticCombinationSupport<ConstantFixtureModelTag,
                           ElementL2Tag,
                           PointwiseEvaluationTag,
                           DGSEMDeviceBackendTag,
                           NoneGradientTreatmentTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

using SupportedHDGDeviceFixtureCombination =
  StaticCombinationSupport<ConstantFixtureModelTag,
                           ElementL2Tag,
                           PointwiseEvaluationTag,
                           HDGDeviceBackendTag,
                           NoneGradientTreatmentTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

using SupportedHDGHybridizedCombination =
  StaticCombinationSupport<ConstantFixtureModelTag,
                           HybridizedTraceTag,
                           GlobalLinearSolveTag,
                           HDGCPUBackendTag,
                           NoneGradientTreatmentTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

using SupportedBrokenDGProbeCombination =
  StaticCombinationSupport<BrokenDGContractProbeModelTag,
                           ElementL2Tag,
                           GlobalLinearSolveTag,
                           DGCPUBackendTag,
                           ConservativeStateBrokenDGTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

using UnsupportedDGHybridizedCombination =
  StaticCombinationSupport<ConstantFixtureModelTag,
                           HybridizedTraceTag,
                           GlobalLinearSolveTag,
                           DGCPUBackendTag,
                           NoneGradientTreatmentTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

using UnsupportedSolveCombination =
  StaticCombinationSupport<ZeroFixtureModelTag,
                           ElementL2Tag,
                           GlobalLinearSolveTag,
                           DGCPUBackendTag,
                           NoneGradientTreatmentTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

using UnsupportedTimeCombination =
  StaticCombinationSupport<ConstantFixtureModelTag,
                           ElementL2Tag,
                           PointwiseEvaluationTag,
                           DGSEMCPUBackendTag,
                           NoneGradientTreatmentTag,
                           LaggedStepTag,
                           EntropyNeutralTag>;

using UnsupportedEntropyCombination =
  StaticCombinationSupport<ConstantFixtureModelTag,
                           ElementL2Tag,
                           PointwiseEvaluationTag,
                           HDGCPUBackendTag,
                           NoneGradientTreatmentTag,
                           FrozenRKStageTag,
                           EntropyStableTag>;

static_assert(SupportedCpuFixtureCombination::value,
              "DG CPU fixture combination must be supported");
static_assert(SupportedDeviceFixtureCombination::value,
              "DGSEM device fixture combination must be supported");
static_assert(SupportedHDGDeviceFixtureCombination::value,
              "HDG device fixture combination must be supported");
static_assert(SupportedHDGHybridizedCombination::value,
              "HDG hybridized fixture combination must be supported");
static_assert(SupportedBrokenDGProbeCombination::value,
              "exact BrokenDG probe combination must be supported");
static_assert(!UnsupportedDGHybridizedCombination::value,
              "DG must not claim the HDG hybridized fixture combination");
static_assert(!UnsupportedSolveCombination::value,
              "unsupported solve combinations must fail at compile time");
static_assert(!UnsupportedTimeCombination::value,
              "unsupported time combinations must fail at compile time");
static_assert(!UnsupportedEntropyCombination::value,
              "unsupported entropy combinations must fail at compile time");

} // namespace regularization
} // namespace dg_igr

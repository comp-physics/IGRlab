#pragma once

#include "physics/regularization/axes.hpp"
#include "physics/regularization/traits.hpp"

#include <type_traits>

namespace dg_igr
{
namespace regularization
{

/** Compile-time support matrix. Model, discretization, auxiliary solve, and
 * backend are separate template parameters; time and entropy are compatibility
 * parameters and cannot be silently inferred from any model axis. */
template <typename ModelTag,
          typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename TimeTag,
          typename EntropyTag>
struct StaticCombinationSupport
  : std::integral_constant<
      bool,
      HasRegularizationTraits<ModelTag>::value &&
        std::is_same<TimeTag, FrozenRKStageTag>::value &&
        std::is_same<EntropyTag, EntropyNeutralTag>::value &&
        ((std::is_same<DiscretizationTag, ElementL2Tag>::value &&
          std::is_same<SolutionTag, PointwiseEvaluationTag>::value &&
          (std::is_same<BackendTag, MFEMCpuBackendTag>::value ||
           std::is_same<BackendTag, MFEMDeviceBackendTag>::value)) ||
         (std::is_same<ModelTag, ConstantFixtureModelTag>::value &&
          std::is_same<DiscretizationTag, TraceL2Tag>::value &&
          std::is_same<SolutionTag, LocalEliminationTag>::value &&
          std::is_same<BackendTag, MFEMCpuBackendTag>::value) ||
         (std::is_same<ModelTag, ConstantFixtureModelTag>::value &&
          std::is_same<DiscretizationTag, HybridizedTraceTag>::value &&
          std::is_same<SolutionTag, GlobalLinearSolveTag>::value &&
          std::is_same<BackendTag, MFEMCpuBackendTag>::value))>
{};

using SupportedCpuFixtureCombination =
  StaticCombinationSupport<ZeroFixtureModelTag,
                           ElementL2Tag,
                           PointwiseEvaluationTag,
                           MFEMCpuBackendTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

using SupportedDeviceFixtureCombination =
  StaticCombinationSupport<ConstantFixtureModelTag,
                           ElementL2Tag,
                           PointwiseEvaluationTag,
                           MFEMDeviceBackendTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

using UnsupportedSolveCombination =
  StaticCombinationSupport<ZeroFixtureModelTag,
                           ElementL2Tag,
                           GlobalLinearSolveTag,
                           MFEMCpuBackendTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

using UnsupportedTimeCombination =
  StaticCombinationSupport<ConstantFixtureModelTag,
                           ElementL2Tag,
                           PointwiseEvaluationTag,
                           MFEMCpuBackendTag,
                           LaggedStepTag,
                           EntropyNeutralTag>;

using UnsupportedEntropyCombination =
  StaticCombinationSupport<ConstantFixtureModelTag,
                           ElementL2Tag,
                           PointwiseEvaluationTag,
                           MFEMCpuBackendTag,
                           FrozenRKStageTag,
                           EntropyStableTag>;

static_assert(SupportedCpuFixtureCombination::value,
              "baseline CPU fixture combination must be supported");
static_assert(SupportedDeviceFixtureCombination::value,
              "static device policy must be available independently");
static_assert(!UnsupportedSolveCombination::value,
              "unsupported solve combinations must fail at compile time");
static_assert(!UnsupportedTimeCombination::value,
              "unsupported time combinations must fail at compile time");
static_assert(!UnsupportedEntropyCombination::value,
              "unsupported entropy combinations must fail at compile time");

} // namespace regularization
} // namespace dg_igr

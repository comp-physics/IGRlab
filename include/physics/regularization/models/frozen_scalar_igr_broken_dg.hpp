#pragma once

#include "igrlab/compat.hpp"

#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include "physics/regularization/auxiliary_field.hpp"
#include "physics/regularization/combination.hpp"
#include "physics/regularization/traits.hpp"

#include <type_traits>

namespace igrlab
{
namespace regularization
{

/** Opt-in physical identity for the frozen-stage scalar IGR model. */
struct FrozenScalarIGRBrokenDGModelTag
{
};

struct FrozenScalarIGRParameters
{
  mfem::real_t alpha = static_cast<mfem::real_t>(0.0);
  mfem::real_t solver_density_floor = static_cast<mfem::real_t>(0.0);
  mfem::real_t flux_density_floor = static_cast<mfem::real_t>(0.0);
  bool sigma_aware_characteristic_speed = false;
};

constexpr int FrozenScalarIGREulerComponents(int dimension)
{
  return dimension + 2;
}

inline constexpr AuxiliaryFieldIdentity kFrozenScalarIGRSigmaIdentity{
  0x4653494752534947ULL,
  0u,
  0u};

inline constexpr AuxiliaryFieldCapabilities FrozenScalarIGRSigmaCapabilities()
{
  return {CapabilityBit(AuxiliaryCapability::StageLocal)};
}

inline ScalarAuxiliaryFieldDescriptor
MakeFrozenScalarIGRSigmaDescriptor(mfem::Vector& storage,
                                   int entity_count,
                                   int dimension,
                                   AuxiliaryFieldProvenance provenance)
{
  return {storage,
          entity_count,
          dimension,
          AuxiliaryFieldOwnership::BorrowedReadOnly,
          provenance,
          kFrozenScalarIGRSigmaIdentity,
          AuxiliaryRestartKind::ColdRecomputed,
          FrozenScalarIGRSigmaCapabilities()};
}

template <>
struct RegularizationOperationRequirements<FrozenScalarIGRBrokenDGModelTag,
                                           AuxiliaryCoefficientOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         false,
                                                         false,
                                                         false,
                                                         false};
};

template <>
struct RegularizationOperationRequirements<FrozenScalarIGRBrokenDGModelTag,
                                           AuxiliarySourceOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         true,
                                                         false,
                                                         false,
                                                         false,
                                                         false};
};

template <>
struct RegularizationOperationRequirements<FrozenScalarIGRBrokenDGModelTag,
                                           VolumeCorrectionOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         true,
                                                         false,
                                                         false,
                                                         false};
};

template <>
struct RegularizationOperationRequirements<FrozenScalarIGRBrokenDGModelTag,
                                           NormalCorrectionOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         true,
                                                         false,
                                                         true,
                                                         false};
};

template <>
struct RegularizationOperationRequirements<FrozenScalarIGRBrokenDGModelTag,
                                           CharacteristicBoundsOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         true,
                                                         false,
                                                         true,
                                                         true};
};

template <> struct RegularizationTraits<FrozenScalarIGRBrokenDGModelTag>
{
  static constexpr bool available = true;
  static constexpr RegularizationCapabilities
    capabilities{true, true, true, true, false, true, false};

  static ModelPreflightResult
  ModelPreflight(RegularizationOperationKind operation,
                 const EvaluatedPointBatchMetadata& metadata)
  {
    if (metadata.dimension < 1 || metadata.dimension > 3)
    {
      return {};
    }

    const int state_components =
      FrozenScalarIGREulerComponents(metadata.dimension);
    if (metadata.state.component_extent != state_components)
    {
      return {false,
              "frozen scalar IGR requires Euler state extent dimension+2"};
    }

    if (operation == RegularizationOperationKind::AuxiliarySource &&
        metadata.conservative_state_gradient.component_extent !=
          state_components * metadata.dimension)
    {
      return {
        false,
        "frozen scalar IGR source requires the conservative Euler gradient"};
    }

    const bool requires_auxiliary =
      operation == RegularizationOperationKind::VolumeCorrection ||
      operation == RegularizationOperationKind::NormalCorrection ||
      operation == RegularizationOperationKind::CharacteristicBounds;
    if (requires_auxiliary && metadata.auxiliary.component_extent != 1)
    {
      return {false, "frozen scalar IGR requires scalar sigma"};
    }

    const bool requires_normal =
      operation == RegularizationOperationKind::NormalCorrection ||
      operation == RegularizationOperationKind::CharacteristicBounds;
    if (requires_normal &&
        metadata.normal.component_extent != metadata.dimension)
    {
      return {false,
              "frozen scalar IGR face operations require a dimension-vector "
              "normal"};
    }

    if (operation == RegularizationOperationKind::CharacteristicBounds &&
        metadata.base_characteristic_speed.component_extent != 1)
    {
      return {false,
              "frozen scalar IGR bounds require scalar base speed"};
    }
    return {};
  }

  MFEM_HOST_DEVICE static void
  AuxiliaryCoefficient(const RegularizationDevicePolicy& policy,
                       const RegularizationPointInput& point,
                       mfem::real_t* coefficient)
  {
    coefficient[0] =
      frozen_scalar_igr_algebra::AlphaOverDensity(policy.coefficient,
                                                  point.State(0),
                                                  policy.source);
  }

  MFEM_HOST_DEVICE static void
  AuxiliarySource(const RegularizationDevicePolicy& policy,
                  const RegularizationPointInput& point,
                  mfem::real_t* source)
  {
    source[0] = frozen_scalar_igr_algebra::SourceFromConservativeStateGradient(
      point.dimension,
      point.state,
      1,
      point.conservative_state_gradient,
      1,
      point.state_components,
      policy.coefficient,
      policy.source);
  }

  MFEM_HOST_DEVICE static void
  AddVolumeCorrection(const RegularizationDevicePolicy& policy,
                      const RegularizationPointInput& point,
                      mfem::real_t* volume_correction)
  {
    frozen_scalar_igr_algebra::AddTensorSigmaFlux(point.dimension,
                                                  point.state,
                                                  1,
                                                  point.Auxiliary(0),
                                                  policy.volume_scale,
                                                  volume_correction,
                                                  point.dimension,
                                                  1);
  }

  MFEM_HOST_DEVICE static void
  AddNormalCorrection(const RegularizationDevicePolicy& policy,
                      const RegularizationPointInput& point,
                      mfem::real_t* normal_correction)
  {
    frozen_scalar_igr_algebra::AddNormalSigmaFlux(point.dimension,
                                                  point.state,
                                                  1,
                                                  point.normal,
                                                  1,
                                                  point.Auxiliary(0),
                                                  policy.volume_scale,
                                                  normal_correction,
                                                  1);
  }

  MFEM_HOST_DEVICE static CharacteristicBoundsResult
  CharacteristicBounds(const RegularizationDevicePolicy& policy,
                       const RegularizationPointInput& point)
  {
    const mfem::real_t sigma =
      policy.wall_scale == static_cast<mfem::real_t>(0.0)
        ? static_cast<mfem::real_t>(0.0)
        : point.Auxiliary(0);
    const mfem::real_t speed =
      frozen_scalar_igr_algebra::FluxCharacteristicSpeed(
        point.BaseCharacteristicSpeed(),
        point.State(0),
        sigma,
        policy.volume_scale);
    return {-speed, speed};
  }
};

template <typename DiscretizationTag,
          typename SolutionTag,
          typename BackendTag,
          typename GradientTag,
          typename TimeTag,
          typename EntropyTag>
struct ModelCombinationSupport<FrozenScalarIGRBrokenDGModelTag,
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

using FrozenScalarIGRBrokenDGCPUCombination =
  StaticCombinationSupport<FrozenScalarIGRBrokenDGModelTag,
                           ElementL2Tag,
                           GlobalLinearSolveTag,
                           DGCPUBackendTag,
                           ConservativeStateBrokenDGTag,
                           FrozenRKStageTag,
                           EntropyNeutralTag>;

static_assert(FrozenScalarIGRBrokenDGCPUCombination::value,
              "the exact frozen scalar BrokenDG CPU combination must bind");
static_assert(
  RegularizationOperationRequirements<FrozenScalarIGRBrokenDGModelTag,
                                      AuxiliaryCoefficientOperationTag>::value
      .state &&
    !RegularizationOperationRequirements<FrozenScalarIGRBrokenDGModelTag,
                                         AuxiliaryCoefficientOperationTag>::
      value.auxiliary,
  "coefficient requires evaluated state only");
static_assert(
  RegularizationOperationRequirements<FrozenScalarIGRBrokenDGModelTag,
                                      AuxiliarySourceOperationTag>::value
      .conservative_state_gradient &&
    !RegularizationOperationRequirements<FrozenScalarIGRBrokenDGModelTag,
                                         AuxiliarySourceOperationTag>::value
       .auxiliary,
  "source requires evaluated state and conservative gradient only");
static_assert(
  RegularizationOperationRequirements<FrozenScalarIGRBrokenDGModelTag,
                                      CharacteristicBoundsOperationTag>::value
      .base_characteristic_speed,
  "bounds require a per-point base characteristic speed");
static_assert(HasRegularizationTraits<FrozenScalarIGRBrokenDGModelTag>::value,
              "frozen scalar IGR traits must satisfy the static contract");
static_assert(
  !RegularizationTraits<FrozenScalarIGRBrokenDGModelTag>::capabilities.wall_law,
  "frozen scalar IGR must not advertise wall physics");
static_assert(
  !RegularizationTraits<FrozenScalarIGRBrokenDGModelTag>::capabilities
     .diagnostics,
  "frozen scalar IGR must not advertise diagnostics");
static_assert(!detail::HasWallHook<FrozenScalarIGRBrokenDGModelTag>::value,
              "the disabled frozen scalar IGR wall hook must be absent");
static_assert(
  !detail::HasDiagnosticsHook<FrozenScalarIGRBrokenDGModelTag>::value,
  "the disabled frozen scalar IGR diagnostics hook must be absent");
static_assert(
  std::is_trivially_copyable<FrozenScalarIGRBrokenDGModelTag>::value,
  "frozen scalar IGR model identity must remain static POD state");
static_assert(!std::is_polymorphic<FrozenScalarIGRBrokenDGModelTag>::value,
              "frozen scalar IGR model identity must not be polymorphic");

} // namespace regularization
} // namespace igrlab

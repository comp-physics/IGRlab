#pragma once

#include "physics/regularization/axes.hpp"
#include "physics/regularization/evaluated_field.hpp"

#include "mfem.hpp"

#include <type_traits>
#include <utility>

namespace dg_igr
{
namespace regularization
{

struct RegularizationCapabilities
{
  bool auxiliary_coefficient = false;
  bool auxiliary_source = false;
  bool volume_correction = false;
  bool normal_correction = false;
  bool wall_law = false;
  bool characteristic_bounds = false;
  bool diagnostics = false;
};

/** POD policy copied into an MFEM kernel closure. It deliberately contains no
 * owning handle, polymorphic object, or runtime model/operation discriminator.
 */
struct RegularizationDevicePolicy
{
  mfem::real_t coefficient = 0.0;
  mfem::real_t source = 0.0;
  mfem::real_t volume_scale = 0.0;
  mfem::real_t wall_scale = 0.0;
  mfem::real_t characteristic_speed = 0.0;
};

struct RegularizationPointInput
{
  const mfem::real_t* state = nullptr;
  const mfem::real_t* auxiliary = nullptr;
  const mfem::real_t* trace = nullptr;
  const mfem::real_t* conservative_state_gradient = nullptr;
  const mfem::real_t* normal = nullptr;
  int state_components = 0;
  int auxiliary_components = 0;
  int trace_components = 0;
  int correction_components = 0;
  int dimension = 0;
  const mfem::real_t* base_characteristic_speed = nullptr;

  MFEM_HOST_DEVICE mfem::real_t State(int component) const
  {
    return state[component];
  }
  MFEM_HOST_DEVICE mfem::real_t Auxiliary(int component) const
  {
    return auxiliary[component];
  }
  MFEM_HOST_DEVICE mfem::real_t Trace(int component) const
  {
    return trace[component];
  }
  MFEM_HOST_DEVICE mfem::real_t ConservativeGradient(int component,
                                                     int direction) const
  {
    return conservative_state_gradient[component +
                                       state_components * direction];
  }
  MFEM_HOST_DEVICE mfem::real_t Normal(int direction) const
  {
    return normal[direction];
  }
  MFEM_HOST_DEVICE mfem::real_t BaseCharacteristicSpeed() const
  {
    return base_characteristic_speed[0];
  }
};

struct CharacteristicBoundsResult
{
  mfem::real_t lower = 0.0;
  mfem::real_t upper = 0.0;
};

struct DiagnosticResult
{
  mfem::real_t source_magnitude = 0.0;
  mfem::real_t correction_l1 = 0.0;
  mfem::real_t wall_magnitude = 0.0;
};

static_assert(std::is_trivially_copyable<RegularizationDevicePolicy>::value,
              "regularization device policy must be trivially copyable");
static_assert(std::is_standard_layout<RegularizationDevicePolicy>::value,
              "regularization device policy must have standard layout");
static_assert(std::is_trivially_copyable<RegularizationPointInput>::value,
              "regularization point input must be trivially copyable");
static_assert(std::is_standard_layout<RegularizationPointInput>::value,
              "regularization point input must have standard layout");

template <typename ModelTag> struct RegularizationTraits;

template <typename ModelTag, typename OperationTag>
struct RegularizationOperationRequirements;

template <typename ModelTag>
struct RegularizationOperationRequirements<ModelTag,
                                           AuxiliaryCoefficientOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         true,
                                                         false,
                                                         false};
};

template <typename ModelTag>
struct RegularizationOperationRequirements<ModelTag,
                                           AuxiliarySourceOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         true,
                                                         false,
                                                         false};
};

template <typename ModelTag>
struct RegularizationOperationRequirements<ModelTag,
                                           VolumeCorrectionOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         true,
                                                         false,
                                                         false};
};

template <typename ModelTag>
struct RegularizationOperationRequirements<ModelTag,
                                           NormalCorrectionOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         true,
                                                         false,
                                                         true};
};

template <typename ModelTag>
struct RegularizationOperationRequirements<ModelTag, WallLawOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         false,
                                                         true,
                                                         false};
};

template <typename ModelTag>
struct RegularizationOperationRequirements<ModelTag,
                                           CharacteristicBoundsOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         false,
                                                         false,
                                                         true};
};

template <typename ModelTag>
struct RegularizationOperationRequirements<ModelTag, DiagnosticsOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         true,
                                                         true,
                                                         false};
};

template <>
struct RegularizationOperationRequirements<BrokenDGContractProbeModelTag,
                                           AuxiliarySourceOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         true,
                                                         true,
                                                         false,
                                                         false};
};

template <>
struct RegularizationOperationRequirements<BrokenDGContractProbeModelTag,
                                           CharacteristicBoundsOperationTag>
{
  static constexpr EvaluatedPointFieldRequirements value{true,
                                                         false,
                                                         false,
                                                         false,
                                                         true,
                                                         true};
};

namespace detail
{

MFEM_HOST_DEVICE inline mfem::real_t AbsoluteValue(mfem::real_t value)
{
  return value < 0.0 ? -value : value;
}

MFEM_HOST_DEVICE inline mfem::real_t FixtureComponentScale(int component)
{
  return static_cast<mfem::real_t>(component + 1);
}

inline bool IsVolumeOperation(RegularizationOperationKind operation)
{
  return operation == RegularizationOperationKind::AuxiliaryCoefficient ||
         operation == RegularizationOperationKind::AuxiliarySource ||
         operation == RegularizationOperationKind::VolumeCorrection;
}

inline bool IsFaceOperation(RegularizationOperationKind operation)
{
  return operation == RegularizationOperationKind::NormalCorrection ||
         operation == RegularizationOperationKind::CharacteristicBounds;
}

} // namespace detail

/** A neutral fixture whose complete regularization contribution is zero. */
template <> struct RegularizationTraits<ZeroFixtureModelTag>
{
  static constexpr bool available = true;
  static constexpr RegularizationCapabilities
    capabilities{true, true, true, true, true, true, true};

  static ModelPreflightResult ModelPreflight(RegularizationOperationKind,
                                             const EvaluatedPointBatchMetadata&)
  {
    return {};
  }

  MFEM_HOST_DEVICE static void
  AuxiliaryCoefficient(const RegularizationDevicePolicy&,
                       const RegularizationPointInput& point,
                       mfem::real_t* coefficient)
  {
    for (int component = 0; component < point.auxiliary_components; ++component)
    {
      coefficient[component] = 0.0;
    }
  }

  MFEM_HOST_DEVICE static void
  AuxiliarySource(const RegularizationDevicePolicy&,
                  const RegularizationPointInput& point,
                  mfem::real_t* source)
  {
    for (int component = 0; component < point.auxiliary_components; ++component)
    {
      source[component] = 0.0;
    }
  }

  MFEM_HOST_DEVICE static void
  AddVolumeCorrection(const RegularizationDevicePolicy&,
                      const RegularizationPointInput&,
                      mfem::real_t*)
  {
  }

  MFEM_HOST_DEVICE static void
  AddNormalCorrection(const RegularizationDevicePolicy&,
                      const RegularizationPointInput&,
                      mfem::real_t*)
  {
  }

  MFEM_HOST_DEVICE static void ApplyWallLaw(const RegularizationDevicePolicy&,
                                            const RegularizationPointInput&,
                                            mfem::real_t*)
  {
  }

  MFEM_HOST_DEVICE static CharacteristicBoundsResult
  CharacteristicBounds(const RegularizationDevicePolicy&,
                       const RegularizationPointInput&)
  {
    return {};
  }

  MFEM_HOST_DEVICE static DiagnosticResult
  Diagnostics(const RegularizationDevicePolicy&,
              const RegularizationPointInput&)
  {
    return {};
  }
};

/** A deterministic algebraic fixture used to exercise every kernel contract.
 * It is intentionally not a physical regularization equation.
 */
template <> struct RegularizationTraits<ConstantFixtureModelTag>
{
  static constexpr bool available = true;
  static constexpr RegularizationCapabilities
    capabilities{true, true, true, true, true, true, true};

  static ModelPreflightResult ModelPreflight(RegularizationOperationKind,
                                             const EvaluatedPointBatchMetadata&)
  {
    return {};
  }

  MFEM_HOST_DEVICE static mfem::real_t
  ComponentCorrection(const RegularizationDevicePolicy& policy,
                      const RegularizationPointInput& point,
                      int component)
  {
    return detail::FixtureComponentScale(component) * policy.volume_scale *
           point.Auxiliary(0);
  }

  MFEM_HOST_DEVICE static mfem::real_t
  ComponentWallFlux(const RegularizationDevicePolicy& policy,
                    const RegularizationPointInput& point,
                    int component)
  {
    return detail::FixtureComponentScale(component) * policy.wall_scale *
           (point.Trace(0) - point.State(0));
  }

  MFEM_HOST_DEVICE static void
  AuxiliaryCoefficient(const RegularizationDevicePolicy& policy,
                       const RegularizationPointInput& point,
                       mfem::real_t* coefficient)
  {
    for (int component = 0; component < point.auxiliary_components; ++component)
    {
      coefficient[component] = policy.coefficient;
    }
  }

  MFEM_HOST_DEVICE static void
  AuxiliarySource(const RegularizationDevicePolicy& policy,
                  const RegularizationPointInput& point,
                  mfem::real_t* source)
  {
    for (int component = 0; component < point.auxiliary_components; ++component)
    {
      source[component] = policy.source;
    }
  }

  MFEM_HOST_DEVICE static void
  AddVolumeCorrection(const RegularizationDevicePolicy& policy,
                      const RegularizationPointInput& point,
                      mfem::real_t* volume_correction)
  {
    for (int component = 0; component < point.correction_components;
         ++component)
    {
      const mfem::real_t correction =
        ComponentCorrection(policy, point, component);
      for (int direction = 0; direction < point.dimension; ++direction)
      {
        volume_correction[direction + point.dimension * component] +=
          correction;
      }
    }
  }

  MFEM_HOST_DEVICE static void
  AddNormalCorrection(const RegularizationDevicePolicy& policy,
                      const RegularizationPointInput& point,
                      mfem::real_t* normal_correction)
  {
    mfem::real_t normal_sum = 0.0;
    for (int direction = 0; direction < point.dimension; ++direction)
    {
      normal_sum += point.Normal(direction);
    }
    for (int component = 0; component < point.correction_components;
         ++component)
    {
      normal_correction[component] +=
        ComponentCorrection(policy, point, component) * normal_sum;
    }
  }

  MFEM_HOST_DEVICE static void
  ApplyWallLaw(const RegularizationDevicePolicy& policy,
               const RegularizationPointInput& point,
               mfem::real_t* wall_flux)
  {
    for (int component = 0; component < point.correction_components;
         ++component)
    {
      wall_flux[component] += ComponentWallFlux(policy, point, component);
    }
  }

  MFEM_HOST_DEVICE static CharacteristicBoundsResult
  CharacteristicBounds(const RegularizationDevicePolicy& policy,
                       const RegularizationPointInput&)
  {
    const mfem::real_t speed =
      detail::AbsoluteValue(policy.characteristic_speed);
    return {-speed, speed};
  }

  MFEM_HOST_DEVICE static DiagnosticResult
  Diagnostics(const RegularizationDevicePolicy& policy,
              const RegularizationPointInput& point)
  {
    mfem::real_t correction_l1 = 0.0;
    mfem::real_t wall_l1 = 0.0;
    for (int component = 0; component < point.correction_components;
         ++component)
    {
      const mfem::real_t correction =
        detail::AbsoluteValue(ComponentCorrection(policy, point, component));
      correction_l1 += static_cast<mfem::real_t>(point.dimension) * correction;
      wall_l1 +=
        detail::AbsoluteValue(ComponentWallFlux(policy, point, component));
    }
    return {detail::AbsoluteValue(policy.source), correction_l1, wall_l1};
  }
};

/** Non-physical probe for the future conservative-gradient BrokenDG adapter. */
template <> struct RegularizationTraits<BrokenDGContractProbeModelTag>
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
      return {false, "BrokenDG probe dimension must be in [1,3]"};
    }
    const int state_components = metadata.dimension + 2;
    if (metadata.state.component_extent != state_components)
    {
      return {false, "BrokenDG probe requires Euler state extent dimension+2"};
    }
    if (detail::IsVolumeOperation(operation) &&
        metadata.site != EvaluationSiteKind::Volume)
    {
      return {false, "coefficient/source/volume require a volume site"};
    }
    if (detail::IsFaceOperation(operation) &&
        metadata.site != EvaluationSiteKind::Face)
    {
      return {false, "normal/bounds require a face site"};
    }
    const bool requires_auxiliary =
      operation == RegularizationOperationKind::AuxiliaryCoefficient ||
      operation == RegularizationOperationKind::AuxiliarySource ||
      operation == RegularizationOperationKind::VolumeCorrection ||
      operation == RegularizationOperationKind::NormalCorrection;
    if (requires_auxiliary && metadata.auxiliary.component_extent != 1)
    {
      return {false, "BrokenDG probe requires scalar auxiliary input"};
    }
    if (operation == RegularizationOperationKind::AuxiliarySource &&
        metadata.conservative_state_gradient.component_extent !=
          state_components * metadata.dimension)
    {
      return {false,
              "BrokenDG source requires conservative Euler gradient extent"};
    }
    if (detail::IsFaceOperation(operation) &&
        metadata.normal.component_extent != metadata.dimension)
    {
      return {false,
              "BrokenDG face operations require a dimension-vector normal"};
    }
    return {};
  }

  MFEM_HOST_DEVICE static void
  AuxiliaryCoefficient(const RegularizationDevicePolicy& policy,
                       const RegularizationPointInput&,
                       mfem::real_t* coefficient)
  {
    coefficient[0] = policy.coefficient;
  }

  MFEM_HOST_DEVICE static void
  AuxiliarySource(const RegularizationDevicePolicy& policy,
                  const RegularizationPointInput& point,
                  mfem::real_t* source)
  {
    source[0] = policy.source + point.ConservativeGradient(0, 0);
  }

  MFEM_HOST_DEVICE static void
  AddVolumeCorrection(const RegularizationDevicePolicy& policy,
                      const RegularizationPointInput& point,
                      mfem::real_t* volume_correction)
  {
    for (int component = 0; component < point.correction_components;
         ++component)
    {
      for (int direction = 0; direction < point.dimension; ++direction)
      {
        volume_correction[direction + point.dimension * component] +=
          detail::FixtureComponentScale(component) * policy.volume_scale *
          point.Auxiliary(0);
      }
    }
  }

  MFEM_HOST_DEVICE static void
  AddNormalCorrection(const RegularizationDevicePolicy& policy,
                      const RegularizationPointInput& point,
                      mfem::real_t* normal_correction)
  {
    mfem::real_t normal_sum = 0.0;
    for (int direction = 0; direction < point.dimension; ++direction)
    {
      normal_sum += point.Normal(direction);
    }
    for (int component = 0; component < point.correction_components;
         ++component)
    {
      normal_correction[component] += detail::FixtureComponentScale(component) *
                                      policy.volume_scale * point.Auxiliary(0) *
                                      normal_sum;
    }
  }

  MFEM_HOST_DEVICE static CharacteristicBoundsResult
  CharacteristicBounds(const RegularizationDevicePolicy&,
                       const RegularizationPointInput& point)
  {
    const mfem::real_t speed =
      detail::AbsoluteValue(point.BaseCharacteristicSpeed());
    return {-speed, speed};
  }
};

namespace detail
{

template <typename ModelTag, typename = void>
struct HasAuxiliaryCoefficientHook : std::false_type
{
};

template <typename ModelTag>
struct HasAuxiliaryCoefficientHook<
  ModelTag,
  std::void_t<decltype(RegularizationTraits<ModelTag>::AuxiliaryCoefficient(
    std::declval<const RegularizationDevicePolicy&>(),
    std::declval<const RegularizationPointInput&>(),
    std::declval<mfem::real_t*>()))>>
  : std::is_same<decltype(RegularizationTraits<ModelTag>::AuxiliaryCoefficient(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>(),
                   std::declval<mfem::real_t*>())),
                 void>
{
};

template <typename ModelTag, typename = void>
struct HasAuxiliarySourceHook : std::false_type
{
};

template <typename ModelTag>
struct HasAuxiliarySourceHook<
  ModelTag,
  std::void_t<decltype(RegularizationTraits<ModelTag>::AuxiliarySource(
    std::declval<const RegularizationDevicePolicy&>(),
    std::declval<const RegularizationPointInput&>(),
    std::declval<mfem::real_t*>()))>>
  : std::is_same<decltype(RegularizationTraits<ModelTag>::AuxiliarySource(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>(),
                   std::declval<mfem::real_t*>())),
                 void>
{
};

template <typename ModelTag, typename = void>
struct HasVolumeHook : std::false_type
{
};

template <typename ModelTag>
struct HasVolumeHook<
  ModelTag,
  std::void_t<decltype(RegularizationTraits<ModelTag>::AddVolumeCorrection(
    std::declval<const RegularizationDevicePolicy&>(),
    std::declval<const RegularizationPointInput&>(),
    std::declval<mfem::real_t*>()))>>
  : std::is_same<decltype(RegularizationTraits<ModelTag>::AddVolumeCorrection(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>(),
                   std::declval<mfem::real_t*>())),
                 void>
{
};

template <typename ModelTag, typename = void>
struct HasNormalHook : std::false_type
{
};

template <typename ModelTag>
struct HasNormalHook<
  ModelTag,
  std::void_t<decltype(RegularizationTraits<ModelTag>::AddNormalCorrection(
    std::declval<const RegularizationDevicePolicy&>(),
    std::declval<const RegularizationPointInput&>(),
    std::declval<mfem::real_t*>()))>>
  : std::is_same<decltype(RegularizationTraits<ModelTag>::AddNormalCorrection(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>(),
                   std::declval<mfem::real_t*>())),
                 void>
{
};

template <typename ModelTag, typename = void>
struct HasWallHook : std::false_type
{
};

template <typename ModelTag>
struct HasWallHook<
  ModelTag,
  std::void_t<decltype(RegularizationTraits<ModelTag>::ApplyWallLaw(
    std::declval<const RegularizationDevicePolicy&>(),
    std::declval<const RegularizationPointInput&>(),
    std::declval<mfem::real_t*>()))>>
  : std::is_same<decltype(RegularizationTraits<ModelTag>::ApplyWallLaw(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>(),
                   std::declval<mfem::real_t*>())),
                 void>
{
};

template <typename ModelTag, typename = void>
struct HasBoundsHook : std::false_type
{
};

template <typename ModelTag>
struct HasBoundsHook<
  ModelTag,
  std::void_t<decltype(RegularizationTraits<ModelTag>::CharacteristicBounds(
    std::declval<const RegularizationDevicePolicy&>(),
    std::declval<const RegularizationPointInput&>()))>>
  : std::is_same<decltype(RegularizationTraits<ModelTag>::CharacteristicBounds(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>())),
                 CharacteristicBoundsResult>
{
};

template <typename ModelTag, typename = void>
struct HasDiagnosticsHook : std::false_type
{
};

template <typename ModelTag>
struct HasDiagnosticsHook<
  ModelTag,
  std::void_t<decltype(RegularizationTraits<ModelTag>::Diagnostics(
    std::declval<const RegularizationDevicePolicy&>(),
    std::declval<const RegularizationPointInput&>()))>>
  : std::is_same<decltype(RegularizationTraits<ModelTag>::Diagnostics(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>())),
                 DiagnosticResult>
{
};

template <typename ModelTag, bool Enabled>
struct RequiredAuxiliaryCoefficientHook : std::true_type
{
};

template <typename ModelTag>
struct RequiredAuxiliaryCoefficientHook<ModelTag, true>
  : HasAuxiliaryCoefficientHook<ModelTag>
{
};

template <typename ModelTag, bool Enabled>
struct RequiredAuxiliarySourceHook : std::true_type
{
};

template <typename ModelTag>
struct RequiredAuxiliarySourceHook<ModelTag, true>
  : HasAuxiliarySourceHook<ModelTag>
{
};

template <typename ModelTag, bool Enabled>
struct RequiredVolumeHook : std::true_type
{
};

template <typename ModelTag>
struct RequiredVolumeHook<ModelTag, true> : HasVolumeHook<ModelTag>
{
};

template <typename ModelTag, bool Enabled>
struct RequiredNormalHook : std::true_type
{
};

template <typename ModelTag>
struct RequiredNormalHook<ModelTag, true> : HasNormalHook<ModelTag>
{
};

template <typename ModelTag, bool Enabled>
struct RequiredWallHook : std::true_type
{
};

template <typename ModelTag>
struct RequiredWallHook<ModelTag, true> : HasWallHook<ModelTag>
{
};

template <typename ModelTag, bool Enabled>
struct RequiredBoundsHook : std::true_type
{
};

template <typename ModelTag>
struct RequiredBoundsHook<ModelTag, true> : HasBoundsHook<ModelTag>
{
};

template <typename ModelTag, bool Enabled>
struct RequiredDiagnosticsHook : std::true_type
{
};

template <typename ModelTag>
struct RequiredDiagnosticsHook<ModelTag, true> : HasDiagnosticsHook<ModelTag>
{
};

template <typename ModelTag, typename = void>
struct TraitDiscovery : std::false_type
{
};

template <typename ModelTag>
struct TraitDiscovery<
  ModelTag,
  std::void_t<decltype(RegularizationTraits<ModelTag>::available),
              decltype(RegularizationTraits<ModelTag>::capabilities),
              decltype(RegularizationTraits<ModelTag>::ModelPreflight(
                std::declval<RegularizationOperationKind>(),
                std::declval<const EvaluatedPointBatchMetadata&>()))>>
  : std::integral_constant<
      bool,
      RegularizationTraits<ModelTag>::available &&
        std::is_same<decltype(RegularizationTraits<ModelTag>::ModelPreflight(
                       std::declval<RegularizationOperationKind>(),
                       std::declval<const EvaluatedPointBatchMetadata&>())),
                     ModelPreflightResult>::value &&
        RequiredAuxiliaryCoefficientHook<
          ModelTag,
          RegularizationTraits<ModelTag>::capabilities
            .auxiliary_coefficient>::value &&
        RequiredAuxiliarySourceHook<ModelTag,
                                    RegularizationTraits<ModelTag>::capabilities
                                      .auxiliary_source>::value &&
        RequiredVolumeHook<ModelTag,
                           RegularizationTraits<ModelTag>::capabilities
                             .volume_correction>::value &&
        RequiredNormalHook<ModelTag,
                           RegularizationTraits<ModelTag>::capabilities
                             .normal_correction>::value &&
        RequiredWallHook<
          ModelTag,
          RegularizationTraits<ModelTag>::capabilities.wall_law>::value &&
        RequiredBoundsHook<ModelTag,
                           RegularizationTraits<ModelTag>::capabilities
                             .characteristic_bounds>::value &&
        RequiredDiagnosticsHook<
          ModelTag,
          RegularizationTraits<ModelTag>::capabilities.diagnostics>::value>
{
};

} // namespace detail

template <typename ModelTag>
struct HasRegularizationTraits : detail::TraitDiscovery<ModelTag>
{
};

template <typename ModelTag, typename OperationTag>
constexpr bool OperationCapabilityEnabled()
{
  constexpr RegularizationCapabilities capabilities =
    RegularizationTraits<ModelTag>::capabilities;
  if constexpr (std::is_same<OperationTag,
                             AuxiliaryCoefficientOperationTag>::value)
  {
    return capabilities.auxiliary_coefficient;
  }
  else if constexpr (std::is_same<OperationTag,
                                  AuxiliarySourceOperationTag>::value)
  {
    return capabilities.auxiliary_source;
  }
  else if constexpr (std::is_same<OperationTag,
                                  VolumeCorrectionOperationTag>::value)
  {
    return capabilities.volume_correction;
  }
  else if constexpr (std::is_same<OperationTag,
                                  NormalCorrectionOperationTag>::value)
  {
    return capabilities.normal_correction;
  }
  else if constexpr (std::is_same<OperationTag, WallLawOperationTag>::value)
  {
    return capabilities.wall_law;
  }
  else if constexpr (std::is_same<OperationTag,
                                  CharacteristicBoundsOperationTag>::value)
  {
    return capabilities.characteristic_bounds;
  }
  else
  {
    static_assert(std::is_same<OperationTag, DiagnosticsOperationTag>::value,
                  "unknown regularization operation tag");
    return capabilities.diagnostics;
  }
}

template <typename ModelTag> struct RegularizationTraitContract
{
  static_assert(HasRegularizationTraits<ModelTag>::value,
                "model tag must satisfy its declared capability subset");
};

static_assert(HasRegularizationTraits<ZeroFixtureModelTag>::value,
              "zero fixture traits must be registered");
static_assert(HasRegularizationTraits<ConstantFixtureModelTag>::value,
              "constant fixture traits must be registered");
static_assert(HasRegularizationTraits<BrokenDGContractProbeModelTag>::value,
              "BrokenDG contract probe traits must be registered");
static_assert(!HasRegularizationTraits<UnregisteredFixtureModelTag>::value,
              "unregistered model tags must fail trait discovery");
static_assert(
  !RegularizationTraits<BrokenDGContractProbeModelTag>::capabilities.wall_law,
  "BrokenDG contract probe must not advertise a wall law");
static_assert(!RegularizationTraits<BrokenDGContractProbeModelTag>::capabilities
                 .diagnostics,
              "BrokenDG contract probe must not advertise diagnostics");
static_assert(!detail::HasWallHook<BrokenDGContractProbeModelTag>::value,
              "disabled probe wall hook must be absent");
static_assert(!detail::HasDiagnosticsHook<BrokenDGContractProbeModelTag>::value,
              "disabled probe diagnostics hook must be absent");

} // namespace regularization
} // namespace dg_igr

#pragma once

#include "physics/regularization/axes.hpp"

#include "mfem.hpp"

#include <type_traits>
#include <utility>

namespace dg_igr
{
namespace regularization
{

struct RegularizationCapabilities
{
  bool coefficient_and_source = false;
  bool volume_correction = false;
  bool normal_correction = false;
  bool wall_law = false;
  bool characteristic_bounds = false;
  bool diagnostics = false;
};

constexpr bool HasAllRegularizationCapabilities(
  RegularizationCapabilities capabilities)
{
  return capabilities.coefficient_and_source &&
         capabilities.volume_correction && capabilities.normal_correction &&
         capabilities.wall_law && capabilities.characteristic_bounds &&
         capabilities.diagnostics;
}

/** POD policy copied into an MFEM kernel closure. It deliberately contains no
 * owning handle, polymorphic object, or runtime model discriminator. */
struct RegularizationDevicePolicy
{
  mfem::real_t coefficient = 0.0;
  mfem::real_t source = 0.0;
  mfem::real_t volume_scale = 0.0;
  mfem::real_t wall_scale = 0.0;
  mfem::real_t characteristic_speed = 0.0;
};

/** Non-owning point view passed to a statically selected model trait.
 *
 * Component-major point slices allow a future system model to consume a state
 * vector and emit equation-by-dimension volume corrections plus equation
 * vectors for normal and wall corrections without changing this interface.
 */
struct RegularizationPointInput
{
  const mfem::real_t* state = nullptr;
  const mfem::real_t* auxiliary = nullptr;
  const mfem::real_t* trace = nullptr;
  const mfem::real_t* state_gradient = nullptr;
  const mfem::real_t* normal = nullptr;
  int state_components = 0;
  int auxiliary_components = 0;
  int trace_components = 0;
  int correction_components = 0;
  int dimension = 0;

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
  MFEM_HOST_DEVICE mfem::real_t Gradient(int component, int direction) const
  {
    return state_gradient[component + state_components * direction];
  }
  MFEM_HOST_DEVICE mfem::real_t Normal(int direction) const
  {
    return normal[direction];
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

template <typename ModelTag>
struct RegularizationTraits;

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

} // namespace detail

/** A neutral fixture whose complete regularization contribution is zero. */
template <>
struct RegularizationTraits<ZeroFixtureModelTag>
{
  static constexpr bool available = true;
  static constexpr RegularizationCapabilities capabilities{
    true, true, true, true, true, true};

  MFEM_HOST_DEVICE static void
  CoefficientsAndSource(const RegularizationDevicePolicy&,
                        const RegularizationPointInput& point,
                        mfem::real_t* coefficient,
                        mfem::real_t* source)
  {
    for (int component = 0; component < point.auxiliary_components; ++component)
    {
      coefficient[component] = 0.0;
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

  MFEM_HOST_DEVICE static void
  ApplyWallLaw(const RegularizationDevicePolicy&,
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
 * It is intentionally not a physical regularization equation. */
template <>
struct RegularizationTraits<ConstantFixtureModelTag>
{
  static constexpr bool available = true;
  static constexpr RegularizationCapabilities capabilities{
    true, true, true, true, true, true};

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
  CoefficientsAndSource(const RegularizationDevicePolicy& policy,
                        const RegularizationPointInput& point,
                        mfem::real_t* coefficient,
                        mfem::real_t* source)
  {
    for (int component = 0; component < point.auxiliary_components; ++component)
    {
      coefficient[component] = policy.coefficient;
      source[component] = policy.source;
    }
  }

  MFEM_HOST_DEVICE static void
  AddVolumeCorrection(const RegularizationDevicePolicy& policy,
                      const RegularizationPointInput& point,
                      mfem::real_t* volume_correction)
  {
    for (int component = 0; component < point.correction_components; ++component)
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
    for (int component = 0; component < point.correction_components; ++component)
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
    for (int component = 0; component < point.correction_components; ++component)
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
    for (int component = 0; component < point.correction_components; ++component)
    {
      const mfem::real_t correction = detail::AbsoluteValue(
        ComponentCorrection(policy, point, component));
      correction_l1 += static_cast<mfem::real_t>(point.dimension) * correction;
      wall_l1 += detail::AbsoluteValue(
        ComponentWallFlux(policy, point, component));
    }
    return {detail::AbsoluteValue(policy.source), correction_l1, wall_l1};
  }
};

template <typename ModelTag, typename = void>
struct HasRegularizationTraits : std::false_type
{};

template <typename ModelTag>
struct HasRegularizationTraits<
  ModelTag,
  std::void_t<
    decltype(RegularizationTraits<ModelTag>::available),
    decltype(RegularizationTraits<ModelTag>::capabilities),
    decltype(RegularizationTraits<ModelTag>::CoefficientsAndSource(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>(),
      std::declval<mfem::real_t*>(),
      std::declval<mfem::real_t*>())),
    decltype(RegularizationTraits<ModelTag>::AddVolumeCorrection(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>(),
      std::declval<mfem::real_t*>())),
    decltype(RegularizationTraits<ModelTag>::AddNormalCorrection(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>(),
      std::declval<mfem::real_t*>())),
    decltype(RegularizationTraits<ModelTag>::ApplyWallLaw(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>(),
      std::declval<mfem::real_t*>())),
    decltype(RegularizationTraits<ModelTag>::CharacteristicBounds(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>())),
    decltype(RegularizationTraits<ModelTag>::Diagnostics(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>()))>>
  : std::integral_constant<
      bool,
      RegularizationTraits<ModelTag>::available &&
        HasAllRegularizationCapabilities(
          RegularizationTraits<ModelTag>::capabilities)>
{};

template <typename ModelTag>
struct RegularizationTraitContract
{
  using Traits = RegularizationTraits<ModelTag>;

  static_assert(HasRegularizationTraits<ModelTag>::value,
                "model tag must provide RegularizationTraits");
  static_assert(
    HasAllRegularizationCapabilities(Traits::capabilities),
    "registered models must enable all six required regularization hooks");
  static_assert(
    std::is_same<decltype(Traits::CoefficientsAndSource(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>(),
                   std::declval<mfem::real_t*>(),
                   std::declval<mfem::real_t*>())),
                 void>::value,
    "coefficient/source signature does not satisfy the static contract");
  static_assert(
    std::is_same<decltype(Traits::AddVolumeCorrection(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>(),
                   std::declval<mfem::real_t*>())),
                 void>::value,
    "volume correction signature does not satisfy the static contract");
  static_assert(
    std::is_same<decltype(Traits::AddNormalCorrection(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>(),
                   std::declval<mfem::real_t*>())),
                 void>::value,
    "normal correction signature does not satisfy the static contract");
  static_assert(
    std::is_same<decltype(Traits::ApplyWallLaw(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>(),
                   std::declval<mfem::real_t*>())),
                 void>::value,
    "wall-law signature does not satisfy the static contract");
  static_assert(
    std::is_same<decltype(Traits::CharacteristicBounds(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>())),
                 CharacteristicBoundsResult>::value,
    "characteristic-bound signature does not satisfy the static contract");
  static_assert(
    std::is_same<decltype(Traits::Diagnostics(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>())),
                 DiagnosticResult>::value,
    "diagnostic signature does not satisfy the static contract");
};

static_assert(HasAllRegularizationCapabilities(
                RegularizationTraits<ZeroFixtureModelTag>::capabilities),
              "zero fixture must enable all required hooks");
static_assert(HasAllRegularizationCapabilities(
                RegularizationTraits<ConstantFixtureModelTag>::capabilities),
              "constant fixture must enable all required hooks");
static_assert(HasRegularizationTraits<ZeroFixtureModelTag>::value,
              "zero fixture traits must be registered");
static_assert(HasRegularizationTraits<ConstantFixtureModelTag>::value,
              "constant fixture traits must be registered");
static_assert(!HasRegularizationTraits<UnregisteredFixtureModelTag>::value,
              "unregistered model tags must fail trait discovery");

} // namespace regularization
} // namespace dg_igr

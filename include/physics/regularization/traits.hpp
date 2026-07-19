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

struct RegularizationPointInput
{
  mfem::real_t state = 0.0;
  mfem::real_t auxiliary = 0.0;
  mfem::real_t trace = 0.0;
  mfem::real_t gradient[3] = {0.0, 0.0, 0.0};
  mfem::real_t normal[3] = {0.0, 0.0, 0.0};
  int dimension = 0;
};

struct CoefficientSourceResult
{
  mfem::real_t coefficient = 0.0;
  mfem::real_t source = 0.0;
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

} // namespace detail

/** A neutral fixture whose complete regularization contribution is zero. */
template <>
struct RegularizationTraits<ZeroFixtureModelTag>
{
  static constexpr bool available = true;
  static constexpr RegularizationCapabilities capabilities{
    true, true, true, true, true, true};

  MFEM_HOST_DEVICE static CoefficientSourceResult
  CoefficientsAndSource(const RegularizationDevicePolicy&,
                        const RegularizationPointInput&)
  {
    return {};
  }

  MFEM_HOST_DEVICE static void
  AddVolumeCorrection(const RegularizationDevicePolicy&,
                      const RegularizationPointInput&,
                      mfem::real_t*)
  {
  }

  MFEM_HOST_DEVICE static mfem::real_t
  NormalCorrection(const RegularizationDevicePolicy&,
                   const RegularizationPointInput&)
  {
    return 0.0;
  }

  MFEM_HOST_DEVICE static mfem::real_t
  WallLaw(const RegularizationDevicePolicy&,
          const RegularizationPointInput&)
  {
    return 0.0;
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

  MFEM_HOST_DEVICE static CoefficientSourceResult
  CoefficientsAndSource(const RegularizationDevicePolicy& policy,
                        const RegularizationPointInput&)
  {
    return {policy.coefficient, policy.source};
  }

  MFEM_HOST_DEVICE static void
  AddVolumeCorrection(const RegularizationDevicePolicy& policy,
                      const RegularizationPointInput& point,
                      mfem::real_t* volume_correction)
  {
    for (int d = 0; d < point.dimension; ++d)
    {
      volume_correction[d] += policy.volume_scale * point.auxiliary;
    }
  }

  MFEM_HOST_DEVICE static mfem::real_t
  NormalCorrection(const RegularizationDevicePolicy& policy,
                   const RegularizationPointInput& point)
  {
    mfem::real_t correction[3] = {0.0, 0.0, 0.0};
    AddVolumeCorrection(policy, point, correction);
    mfem::real_t normal_correction = 0.0;
    for (int d = 0; d < point.dimension; ++d)
    {
      normal_correction += correction[d] * point.normal[d];
    }
    return normal_correction;
  }

  MFEM_HOST_DEVICE static mfem::real_t
  WallLaw(const RegularizationDevicePolicy& policy,
          const RegularizationPointInput& point)
  {
    return policy.wall_scale * (point.trace - point.state);
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
    const mfem::real_t component =
      detail::AbsoluteValue(policy.volume_scale * point.auxiliary);
    for (int d = 0; d < point.dimension; ++d)
    {
      correction_l1 += component;
    }
    return {detail::AbsoluteValue(policy.source),
            correction_l1,
            detail::AbsoluteValue(WallLaw(policy, point))};
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
      std::declval<const RegularizationPointInput&>())),
    decltype(RegularizationTraits<ModelTag>::AddVolumeCorrection(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>(),
      std::declval<mfem::real_t*>())),
    decltype(RegularizationTraits<ModelTag>::NormalCorrection(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>())),
    decltype(RegularizationTraits<ModelTag>::WallLaw(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>())),
    decltype(RegularizationTraits<ModelTag>::CharacteristicBounds(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>())),
    decltype(RegularizationTraits<ModelTag>::Diagnostics(
      std::declval<const RegularizationDevicePolicy&>(),
      std::declval<const RegularizationPointInput&>()))>>
  : std::integral_constant<bool, RegularizationTraits<ModelTag>::available>
{};

template <typename ModelTag>
struct RegularizationTraitContract
{
  using Traits = RegularizationTraits<ModelTag>;

  static_assert(HasRegularizationTraits<ModelTag>::value,
                "model tag must provide RegularizationTraits");
  static_assert(
    std::is_same<decltype(Traits::CoefficientsAndSource(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>())),
                 CoefficientSourceResult>::value,
    "coefficient/source signature does not satisfy the static contract");
  static_assert(
    std::is_same<decltype(Traits::AddVolumeCorrection(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>(),
                   std::declval<mfem::real_t*>())),
                 void>::value,
    "volume correction signature does not satisfy the static contract");
  static_assert(
    std::is_same<decltype(Traits::NormalCorrection(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>())),
                 mfem::real_t>::value,
    "normal correction signature does not satisfy the static contract");
  static_assert(
    std::is_same<decltype(Traits::WallLaw(
                   std::declval<const RegularizationDevicePolicy&>(),
                   std::declval<const RegularizationPointInput&>())),
                 mfem::real_t>::value,
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

static_assert(HasRegularizationTraits<ZeroFixtureModelTag>::value,
              "zero fixture traits must be registered");
static_assert(HasRegularizationTraits<ConstantFixtureModelTag>::value,
              "constant fixture traits must be registered");
static_assert(!HasRegularizationTraits<UnregisteredFixtureModelTag>::value,
              "unregistered model tags must fail trait discovery");

} // namespace regularization
} // namespace dg_igr

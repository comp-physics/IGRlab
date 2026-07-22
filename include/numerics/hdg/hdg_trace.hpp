#pragma once

/**
 * @file numerics/hdg/hdg_trace.hpp
 * @brief Compile-time HDG trace-field shapes and tuples.
 */

#include <cstddef>
#include <limits>
#include <type_traits>

namespace dg
{
namespace hdg
{

struct ScalarTrace
{
  static constexpr std::size_t components = 1;
};

template <typename T>
struct IsTraceShape : std::false_type
{};

template <>
struct IsTraceShape<ScalarTrace> : std::true_type
{};

template <std::size_t Components>
struct VectorTrace
{
  static_assert(Components > 0, "an HDG vector trace cannot be empty");
  static constexpr std::size_t components = Components;
};

template <std::size_t Components>
struct IsTraceShape<VectorTrace<Components>> : std::true_type
{};

template <std::size_t Rows, std::size_t Columns>
struct TensorTrace
{
  static_assert(Rows > 0 && Columns > 0,
                "an HDG tensor trace cannot be empty");
  static_assert(Rows <= std::numeric_limits<std::size_t>::max() / Columns,
                "an HDG tensor trace component count overflows size_t");
  static constexpr std::size_t rows = Rows;
  static constexpr std::size_t columns = Columns;
  static constexpr std::size_t components = Rows * Columns;
};

template <std::size_t Rows, std::size_t Columns>
struct IsTraceShape<TensorTrace<Rows, Columns>> : std::true_type
{};

/** Associates a model-owned name tag with one scalar/vector/tensor shape. */
template <typename NameTag, typename Shape>
struct TracePotential
{
  static_assert(IsTraceShape<Shape>::value,
                "an HDG trace potential requires a scalar, vector, or tensor trace shape");
  using name_tag = NameTag;
  using shape = Shape;
  static constexpr std::size_t components = Shape::components;
};

namespace detail
{

template <typename... Tags>
struct AreUnique : std::true_type
{};

template <typename First, typename... Rest>
struct AreUnique<First, Rest...>
    : std::integral_constant<bool,
                             ((!std::is_same<First, Rest>::value) && ...) &&
                                 AreUnique<Rest...>::value>
{};

template <typename T>
struct IsTracePotential : std::false_type
{};

template <typename NameTag, typename Shape>
struct IsTracePotential<TracePotential<NameTag, Shape>>
    : IsTraceShape<Shape>
{};

template <typename... Potentials>
struct TraceComponentCount;

template <>
struct TraceComponentCount<>
{
  static constexpr std::size_t value = 0;
};

template <typename First, typename... Rest>
struct TraceComponentCount<First, Rest...>
{
  static constexpr std::size_t rest = TraceComponentCount<Rest...>::value;
  static_assert(First::components <= std::numeric_limits<std::size_t>::max() -
                                         rest,
                "HDG trace tuple component count overflows size_t");
  static constexpr std::size_t value = First::components + rest;
};

template <std::size_t Index, typename First, typename... Rest>
struct TracePotentialAt : TracePotentialAt<Index - 1, Rest...>
{};

template <typename First, typename... Rest>
struct TracePotentialAt<0, First, Rest...>
{
  using type = First;
};

template <std::size_t Index, typename First, typename... Rest>
struct TraceComponentOffset
{
  static constexpr std::size_t value =
      First::components + TraceComponentOffset<Index - 1, Rest...>::value;
};

template <typename First, typename... Rest>
struct TraceComponentOffset<0, First, Rest...>
{
  static constexpr std::size_t value = 0;
};

} // namespace detail

/** Type-only ordered tuple of independently named trace potentials. */
template <typename... Potentials>
struct TraceTuple
{
  static_assert(sizeof...(Potentials) > 0,
                "an HDG trace tuple needs at least one potential");
  static_assert((detail::IsTracePotential<Potentials>::value && ...),
                "HDG_CONTRACT_TRACE_POTENTIAL: trace tuple requires TracePotential entries");
  static_assert(detail::AreUnique<typename Potentials::name_tag...>::value,
                "HDG_CONTRACT_TRACE_UNIQUE_NAMES: trace potential name tags must be unique");
  static constexpr std::size_t potential_count = sizeof...(Potentials);
  static constexpr std::size_t component_count =
      detail::TraceComponentCount<Potentials...>::value;

  template <std::size_t Index>
  using potential_type =
      typename detail::TracePotentialAt<Index, Potentials...>::type;

  template <std::size_t Index>
  static constexpr std::size_t ComponentOffset() noexcept
  {
    static_assert(Index < potential_count, "trace potential index is invalid");
    return detail::TraceComponentOffset<Index, Potentials...>::value;
  }
};

template <typename T>
struct IsTraceTuple : std::false_type
{};

template <typename... Potentials>
struct IsTraceTuple<TraceTuple<Potentials...>> : std::true_type
{};

static_assert(std::is_trivially_copyable<ScalarTrace>::value,
              "scalar trace tags must be trivially copyable");
static_assert(std::is_trivially_copyable<VectorTrace<1>>::value,
              "vector trace tags must be trivially copyable");
static_assert(std::is_trivially_copyable<TensorTrace<1, 1>>::value,
              "tensor trace tags must be trivially copyable");

} // namespace hdg
} // namespace dg

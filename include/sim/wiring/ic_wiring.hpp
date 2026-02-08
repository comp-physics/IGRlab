#pragma once

/**
 * @file sim/wiring/ic_wiring.hpp
 * @brief Initial-condition coefficient helpers with explicit ownership semantics.
 */

#include <memory>
#include <type_traits>
#include <utility>

#include "mfem.hpp"

namespace dg_igr_examples
{

/**
 * @brief Build an owning IC coefficient from any VectorFunctionCoefficient ctor args.
 *
 * This mirrors the ergonomics of BC wiring helpers by returning the required
 * solver-facing ownership type (`std::unique_ptr<mfem::VectorCoefficient>`).
 */
template <typename... Args>
inline std::unique_ptr<mfem::VectorCoefficient> MakeVectorFunctionIC(Args&&... args)
{
  static_assert(std::is_constructible_v<mfem::VectorFunctionCoefficient, Args...>,
                "MakeVectorFunctionIC arguments must match a "
                "VectorFunctionCoefficient constructor.");
  return std::make_unique<mfem::VectorFunctionCoefficient>(
      std::forward<Args>(args)...);
}

} // namespace dg_igr_examples


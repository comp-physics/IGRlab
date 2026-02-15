#pragma once

/**
 * @file utils/math/constants.hpp
 * @brief Centralized project math constants.
 */

#include "mfem.hpp"

namespace dg_sim::math
{

// Keep constants MFEM-portable for future device kernels.
MFEM_HOST_DEVICE inline constexpr mfem::real_t kPi =
  static_cast<mfem::real_t>(3.141592653589793238462643383279502884L);
MFEM_HOST_DEVICE inline constexpr mfem::real_t kTwoPi =
  static_cast<mfem::real_t>(2.0) * kPi;
MFEM_HOST_DEVICE inline constexpr mfem::real_t kHalfPi =
  static_cast<mfem::real_t>(0.5) * kPi;

} // namespace dg_sim::math

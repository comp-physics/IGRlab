#pragma once

#include "mfem.hpp"

/**
 * @file numerics/limiters/krivodonova.hpp
 * @brief Krivodonova/BDS hierarchical modal limiter for DG.
 * @ingroup numerics_limiters
 */

namespace mfem
{

/**
 * @brief Options for the hierarchical modal limiter (Krivodonova / BDS style).
 */
struct KrivoOptions {
  int    dim        = 2;        ///< spatial dimension (2)
  double gamma      = 1.4;      ///< EOS, only used if you later switch to characteristic space
  int    face_qord  = 0;        ///< quadrature on each face (0 => auto: p+2)
  double inflate    = 1.0;      ///< widen neighbor bounds: [umin-eps, umax+eps]
  bool   conservative_only = true; ///< if true, limit component-wise (robust)
};

/**
 * @brief Element-local hierarchical modal limiter (Krivodonova/BDS) for quads.
 */
void ApplyKrivodonovaHierarchicalLimiter(mfem::ParFiniteElementSpace &vfes,
                                         mfem::ParGridFunction &U,
                                         const KrivoOptions &opt);

} // namespace mfem

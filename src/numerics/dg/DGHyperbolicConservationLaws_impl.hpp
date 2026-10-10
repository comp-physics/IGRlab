#pragma once

#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "physics/bc/detail/euler_boundary_state.hpp"
#include "physics/igr/igr_flux.hpp"
#include "utils/config/env_flags.hpp"

#if !((defined(MFEM_USE_CUDA) && !defined(__CUDACC__)) || \
      (defined(MFEM_USE_HIP) && !defined(__HIPCC__)))
#include "mfem/general/forall.hpp"
#include "mfem/general/reducers.hpp"
#ifndef DG_IGR_HAS_ACCEL_FORALL
#define DG_IGR_HAS_ACCEL_FORALL 1
#endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace mfem
{
namespace dg_igr_detail
{

constexpr int kFrozenSigmaTelemetryWidth = 5;

inline void InitializeFrozenSigmaTelemetry(Array<double>& storage,
                                           int sample_slots)
{
  MFEM_VERIFY(sample_slots >= 0,
              "Frozen-sigma telemetry received a negative slot count.");
  storage.SetSize(sample_slots * kFrozenSigmaTelemetryWidth);
  double* const values = storage.HostWrite();
  for (int slot = 0; slot < sample_slots; ++slot)
  {
    const int offset = slot * kFrozenSigmaTelemetryWidth;
    values[offset + 0] = 0.0;
    values[offset + 1] = 0.0;
    values[offset + 2] = 0.0;
    values[offset + 3] = std::numeric_limits<double>::infinity();
    values[offset + 4] = std::numeric_limits<double>::infinity();
  }
}

} // namespace dg_igr_detail
} // namespace mfem

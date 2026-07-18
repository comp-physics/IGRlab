#pragma once

#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
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
#include <string>
#include <vector>

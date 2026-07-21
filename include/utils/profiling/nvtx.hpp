#pragma once

/**
 * @file utils/profiling/nvtx.hpp
 * @brief Source-compatible name for the core observability range.
 */

#include "utils/profiling/observability.hpp"

namespace dg_igr
{

using NVTXRange = observability::ScopedRegion<observability::DefaultBackend>;

} // namespace dg_igr

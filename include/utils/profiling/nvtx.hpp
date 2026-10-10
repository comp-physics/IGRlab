#pragma once

#include "igrlab/compat.hpp"

/**
 * @file utils/profiling/nvtx.hpp
 * @brief Source-compatible name for the core observability range.
 */

#include "utils/profiling/observability.hpp"

namespace igrlab
{

using NVTXRange = observability::ScopedRegion<observability::DefaultBackend>;

} // namespace igrlab

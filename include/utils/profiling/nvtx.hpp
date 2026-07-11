#pragma once

/**
 * @file utils/profiling/nvtx.hpp
 * @brief Tiny RAII wrapper for optional NVTX ranges.
 */

#if defined(__has_include)
#if __has_include(<nvtx3/nvtx3.hpp>)
#define DG_IGR_HAS_NVTX3_HEADER 1
#include <nvtx3/nvtx3.hpp>
#endif
#endif

namespace dg_igr
{

class NVTXRange
{
public:
#ifdef DG_IGR_HAS_NVTX3_HEADER
  explicit NVTXRange(const char* name) : range_(name) {}
#else
  explicit NVTXRange(const char*) {}
#endif

  NVTXRange(const NVTXRange&) = delete;
  NVTXRange& operator=(const NVTXRange&) = delete;

private:
#ifdef DG_IGR_HAS_NVTX3_HEADER
  nvtx3::scoped_range range_;
#endif
};

} // namespace dg_igr

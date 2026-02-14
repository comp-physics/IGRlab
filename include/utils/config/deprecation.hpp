#pragma once

/**
 * @file utils/config/deprecation.hpp
 * @brief Deprecation helpers for staged API migrations.
 * @ingroup utils_config
 */

#if defined(DG_IGR_ENABLE_DEPRECATION_WARNINGS)
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
#define DG_IGR_DEPRECATED(msg) [[deprecated(msg)]]
#else
#define DG_IGR_DEPRECATED(msg)
#endif
#else
#define DG_IGR_DEPRECATED(msg)
#endif
#else
// Deprecation markers are opt-in to avoid warning noise during migration.
#define DG_IGR_DEPRECATED(msg)
#endif

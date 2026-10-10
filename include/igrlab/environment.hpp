#pragma once

#include "igrlab/compat.hpp"
#include <cstdlib>
#include <cstring>
#include <string>

namespace igrlab
{
// Prefer the canonical spelling while accepting existing runtime scripts.
inline const char* GetEnv(const char* name)
{
  constexpr const char* legacy_prefix = "DG_IGR_";
  constexpr std::size_t prefix_size = 7;
  if (std::strncmp(name, legacy_prefix, prefix_size) == 0)
  {
    const std::string canonical = std::string("IGRLAB_") + (name + prefix_size);
    if (const char* value = std::getenv(canonical.c_str()))
    {
      return value;
    }
  }
  return std::getenv(name);
}
} // namespace igrlab

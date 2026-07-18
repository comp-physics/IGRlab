#pragma once

#include <cstdlib>
#include <string>

namespace dg_igr
{

inline bool EnvFlagEnabled(const char* name)
{
  const char* raw = std::getenv(name);
  if (!raw)
  {
    return false;
  }
  const std::string value(raw);
  return !(value.empty() || value == "0" || value == "false" ||
           value == "FALSE" || value == "off" || value == "OFF");
}

inline bool EnvFlagEnabledDefault(const char* name, bool default_value)
{
  const char* raw = std::getenv(name);
  if (!raw)
  {
    return default_value;
  }
  return EnvFlagEnabled(name);
}

} // namespace dg_igr

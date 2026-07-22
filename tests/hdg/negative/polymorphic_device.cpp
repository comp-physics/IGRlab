#include "negative_support.hpp"

struct BadDevicePolicy
{
  virtual ~BadDevicePolicy() = default;
};
using BadSpec = dg::hdg::KernelSpec<hdg_negative::GoodModel,
                                    hdg_negative::GoodTrace,
                                    hdg_negative::GoodStabilization,
                                    hdg_negative::GoodTime,
                                    hdg_negative::GoodBackend,
                                    hdg_negative::GoodPrecision,
                                    BadDevicePolicy>;
static_assert(sizeof(BadSpec) > 0, "must instantiate polymorphic device policy");

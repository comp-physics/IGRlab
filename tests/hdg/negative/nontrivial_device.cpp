#include "negative_support.hpp"

struct BadDevicePolicy
{
  ~BadDevicePolicy() {}
  double coefficient = 0.0;
};
using BadSpec = dg::hdg::KernelSpec<hdg_negative::GoodModel,
                                    hdg_negative::GoodTrace,
                                    hdg_negative::GoodStabilization,
                                    hdg_negative::GoodTime,
                                    hdg_negative::GoodBackend,
                                    hdg_negative::GoodPrecision,
                                    BadDevicePolicy>;
static_assert(sizeof(BadSpec) > 0, "force nontrivial device policy");

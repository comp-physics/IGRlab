#include "negative_support.hpp"

struct BadDevicePolicy
{
public:
  int first = 0;

private:
  int second = 0;
};
static_assert(std::is_trivially_copyable<BadDevicePolicy>::value,
              "case must isolate standard layout");
static_assert(!std::is_standard_layout<BadDevicePolicy>::value,
              "case must be non-standard-layout");
using BadSpec = dg::hdg::KernelSpec<hdg_negative::GoodModel,
                                    hdg_negative::GoodTrace,
                                    hdg_negative::GoodStabilization,
                                    hdg_negative::GoodTime,
                                    hdg_negative::GoodBackend,
                                    hdg_negative::GoodPrecision,
                                    BadDevicePolicy>;
static_assert(sizeof(BadSpec) > 0, "force non-standard-layout device policy");

#include "negative_support.hpp"

struct BadTime
{};
using BadSpec = dg::hdg::KernelSpec<hdg_negative::GoodModel,
                                    hdg_negative::GoodTrace,
                                    hdg_negative::GoodStabilization,
                                    BadTime,
                                    hdg_negative::GoodBackend,
                                    hdg_negative::GoodPrecision,
                                    hdg_negative::DevicePolicy>;
static_assert(sizeof(BadSpec) > 0, "must instantiate invalid time treatment");

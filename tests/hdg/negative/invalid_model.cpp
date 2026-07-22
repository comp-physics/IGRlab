#include "negative_support.hpp"

struct BadModel
{};
using BadSpec = dg::hdg::KernelSpec<BadModel,
                                    hdg_negative::GoodTrace,
                                    hdg_negative::GoodStabilization,
                                    hdg_negative::GoodTime,
                                    hdg_negative::GoodBackend,
                                    hdg_negative::GoodPrecision,
                                    hdg_negative::DevicePolicy>;
static_assert(sizeof(BadSpec) > 0, "must instantiate the invalid model");

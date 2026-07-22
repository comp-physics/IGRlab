#include "negative_support.hpp"

struct BadBackend
{};
using BadSpec = dg::hdg::KernelSpec<hdg_negative::GoodModel,
                                    hdg_negative::GoodTrace,
                                    hdg_negative::GoodStabilization,
                                    hdg_negative::GoodTime,
                                    BadBackend,
                                    hdg_negative::GoodPrecision,
                                    hdg_negative::DevicePolicy>;
static_assert(sizeof(BadSpec) > 0, "must instantiate invalid backend");

#include "negative_support.hpp"

using BadSpec = dg::hdg::KernelSpec<hdg_negative::GoodModel,
                                    int,
                                    hdg_negative::GoodStabilization,
                                    hdg_negative::GoodTime,
                                    hdg_negative::GoodBackend,
                                    hdg_negative::GoodPrecision,
                                    hdg_negative::DevicePolicy>;
static_assert(sizeof(BadSpec) > 0, "force non-TraceTuple KernelSpec");

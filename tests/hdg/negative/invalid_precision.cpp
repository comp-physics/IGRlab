#include "negative_support.hpp"

using BadPrecision =
    dg::hdg::PrecisionPolicy<double, float, double, double, double>;
using BadSpec = dg::hdg::KernelSpec<hdg_negative::GoodModel,
                                    hdg_negative::GoodTrace,
                                    hdg_negative::GoodStabilization,
                                    hdg_negative::GoodTime,
                                    hdg_negative::GoodBackend,
                                    BadPrecision,
                                    hdg_negative::DevicePolicy>;
static_assert(sizeof(BadSpec) > 0, "must instantiate invalid precision");

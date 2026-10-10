#include "negative_support.hpp"

static_assert(sizeof(hdg_negative::GoodSpec) > 0,
              "positive KernelSpec must instantiate");
static_assert(sizeof(hdg_negative::GoodPipeline) > 0,
              "positive Pipeline must instantiate");

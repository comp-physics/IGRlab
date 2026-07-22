#include "negative_support.hpp"

struct NotATracePotential
{};
using BadTrace = dg::hdg::TraceTuple<NotATracePotential>;
static_assert(sizeof(BadTrace) > 0, "must instantiate the invalid trace");

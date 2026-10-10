#include "negative_support.hpp"

using BadTrace =
    dg::hdg::TraceTuple<dg::hdg::TracePotential<hdg_negative::StateName,
                                                dg::hdg::ScalarTrace>,
                        dg::hdg::TracePotential<hdg_negative::StateName,
                                                dg::hdg::VectorTrace<2>>>;
static_assert(sizeof(BadTrace) > 0, "must instantiate duplicate trace tags");

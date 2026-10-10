#include "negative_support.hpp"

struct BadCondensation : hdg_negative::GoodCondensation
{
  virtual ~BadCondensation() = default;
};
using BadPipeline = dg::hdg::Pipeline<hdg_negative::GoodSpec,
                                      hdg_negative::GoodLocal,
                                      BadCondensation,
                                      hdg_negative::GoodAssembly,
                                      hdg_negative::GoodSolver,
                                      hdg_negative::GoodRecovery>;
static_assert(sizeof(BadPipeline) > 0, "force polymorphic condensation phase");

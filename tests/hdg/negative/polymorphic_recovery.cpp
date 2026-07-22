#include "negative_support.hpp"

struct BadRecovery : hdg_negative::GoodRecovery
{
  virtual ~BadRecovery() = default;
};
using BadPipeline = dg::hdg::Pipeline<hdg_negative::GoodSpec,
                                      hdg_negative::GoodLocal,
                                      hdg_negative::GoodCondensation,
                                      hdg_negative::GoodAssembly,
                                      hdg_negative::GoodSolver,
                                      BadRecovery>;
static_assert(sizeof(BadPipeline) > 0, "force polymorphic recovery phase");

#include "negative_support.hpp"

struct BadLocal : hdg_negative::GoodLocal
{
  virtual ~BadLocal() = default;
};
using BadPipeline = dg::hdg::Pipeline<hdg_negative::GoodSpec,
                                      BadLocal,
                                      hdg_negative::GoodCondensation,
                                      hdg_negative::GoodAssembly,
                                      hdg_negative::GoodSolver,
                                      hdg_negative::GoodRecovery>;
static_assert(sizeof(BadPipeline) > 0, "must instantiate polymorphic phase");

#include "negative_support.hpp"

struct BadSolver : hdg_negative::GoodSolver
{
  virtual ~BadSolver() = default;
};
using BadPipeline = dg::hdg::Pipeline<hdg_negative::GoodSpec,
                                      hdg_negative::GoodLocal,
                                      hdg_negative::GoodCondensation,
                                      hdg_negative::GoodAssembly,
                                      BadSolver,
                                      hdg_negative::GoodRecovery>;
static_assert(sizeof(BadPipeline) > 0, "force polymorphic solver phase");

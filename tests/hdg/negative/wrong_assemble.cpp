#include "negative_support.hpp"

struct BadAssembly
{
  void Assemble(int) {}
};
using BadPipeline = dg::hdg::Pipeline<hdg_negative::GoodSpec,
                                      hdg_negative::GoodLocal,
                                      hdg_negative::GoodCondensation,
                                      BadAssembly,
                                      hdg_negative::GoodSolver,
                                      hdg_negative::GoodRecovery>;
static_assert(sizeof(BadPipeline) > 0, "force wrong Assemble");

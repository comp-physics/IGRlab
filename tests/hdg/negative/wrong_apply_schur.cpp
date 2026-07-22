#include "negative_support.hpp"

struct BadCondensation
{
  void Condense(const dg::hdg::CondenseRequest<hdg_negative::GoodSpec>&,
                dg::hdg::CondenseResult<hdg_negative::GoodSpec>&)
  {}
  void ApplySchur(int) {}
};
using BadPipeline = dg::hdg::Pipeline<hdg_negative::GoodSpec,
                                      hdg_negative::GoodLocal,
                                      BadCondensation,
                                      hdg_negative::GoodAssembly,
                                      hdg_negative::GoodSolver,
                                      hdg_negative::GoodRecovery>;
static_assert(sizeof(BadPipeline) > 0, "force wrong ApplySchur");

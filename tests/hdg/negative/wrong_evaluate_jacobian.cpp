#include "negative_support.hpp"

struct BadLocal
{
  void EvaluateResidual(
      const dg::hdg::LocalResidualRequest<hdg_negative::GoodSpec>&,
      dg::hdg::LocalResidualResult<hdg_negative::GoodSpec>&)
  {}
  void EvaluateJacobian(int) {}
};
using BadPipeline = dg::hdg::Pipeline<hdg_negative::GoodSpec,
                                      BadLocal,
                                      hdg_negative::GoodCondensation,
                                      hdg_negative::GoodAssembly,
                                      hdg_negative::GoodSolver,
                                      hdg_negative::GoodRecovery>;
static_assert(sizeof(BadPipeline) > 0, "force wrong EvaluateJacobian");

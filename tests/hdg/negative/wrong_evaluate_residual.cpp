#include "negative_support.hpp"

struct BadLocal
{
  void EvaluateResidual(int) {}
  void EvaluateJacobian(
      const dg::hdg::LocalJacobianRequest<hdg_negative::GoodSpec>&,
      dg::hdg::LocalJacobianResult<hdg_negative::GoodSpec>&)
  {}
};
using BadPipeline = dg::hdg::Pipeline<hdg_negative::GoodSpec,
                                      BadLocal,
                                      hdg_negative::GoodCondensation,
                                      hdg_negative::GoodAssembly,
                                      hdg_negative::GoodSolver,
                                      hdg_negative::GoodRecovery>;
static_assert(sizeof(BadPipeline) > 0, "force wrong EvaluateResidual");

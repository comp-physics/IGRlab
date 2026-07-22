#include "negative_support.hpp"

struct BadLocal
{
  void EvaluateResidual(
      const dg::hdg::LocalResidualRequest<hdg_negative::GoodSpec>&,
      dg::hdg::LocalResidualResult<hdg_negative::GoodSpec>&,
      ...)
  {}
  void EvaluateJacobian(
      const dg::hdg::LocalJacobianRequest<hdg_negative::GoodSpec>&,
      dg::hdg::LocalJacobianResult<hdg_negative::GoodSpec>&)
  {}
};
static_assert(dg::hdg::detail::HasEvaluateResidual<
                  BadLocal,
                  dg::hdg::LocalResidualRequest<hdg_negative::GoodSpec>,
                  dg::hdg::LocalResidualResult<hdg_negative::GoodSpec>>::value,
              "HDG_NEGATIVE_VARIADIC_PHASE_METHOD");
using BadPipeline = dg::hdg::Pipeline<hdg_negative::GoodSpec,
                                      BadLocal,
                                      hdg_negative::GoodCondensation,
                                      hdg_negative::GoodAssembly,
                                      hdg_negative::GoodSolver,
                                      hdg_negative::GoodRecovery>;
static_assert(sizeof(BadPipeline) > 0, "force variadic phase method");

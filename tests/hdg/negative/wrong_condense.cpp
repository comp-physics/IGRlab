#include "negative_support.hpp"

struct BadCondensation
{
  void Condense(int) {}
  void ApplySchur(const dg::hdg::ApplySchurRequest<hdg_negative::GoodSpec>&,
                  dg::hdg::ApplySchurResult<hdg_negative::GoodSpec>&)
  {}
};
using BadPipeline = dg::hdg::Pipeline<hdg_negative::GoodSpec,
                                      hdg_negative::GoodLocal,
                                      BadCondensation,
                                      hdg_negative::GoodAssembly,
                                      hdg_negative::GoodSolver,
                                      hdg_negative::GoodRecovery>;
static_assert(sizeof(BadPipeline) > 0, "force wrong Condense");

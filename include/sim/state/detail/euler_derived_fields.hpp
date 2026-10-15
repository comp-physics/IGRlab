#pragma once

#include "mfem.hpp"
#include <memory>

namespace dg_sim
{
namespace detail
{
struct EulerDerivedFieldPlan;
enum class EulerDerivedField { Pressure, Velocity };

// Cache only invariant FE layout admission. Current state values are always
// reread. False requests the original coefficient projection for the whole call.
bool TryProjectEulerDerivedField(
  const mfem::ParGridFunction& state, mfem::ParGridFunction& output,
  int dim, double gamma, double rho_floor, EulerDerivedField field,
  std::shared_ptr<EulerDerivedFieldPlan>& plan);
}
}

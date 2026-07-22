#pragma once

#include "numerics/hdg/hdg_contract.hpp"

namespace hdg_negative
{

using namespace dg::hdg;

struct GoodModel : ModelTagMarker
{};
struct GoodStabilization : StabilizationTagMarker
{};
struct GoodTime : TimeTreatmentTagMarker
{};
struct StateName;
struct Execution;
struct Memory;
struct Communication;
struct DevicePolicy
{
  double coefficient = 0.0;
};

using GoodTrace = TraceTuple<TracePotential<StateName, ScalarTrace>>;
using GoodBackend = BackendPolicy<Execution, Memory, Communication>;
using GoodPrecision =
    PrecisionPolicy<double, double, double, double, double>;
using GoodSpec = KernelSpec<GoodModel,
                            GoodTrace,
                            GoodStabilization,
                            GoodTime,
                            GoodBackend,
                            GoodPrecision,
                            DevicePolicy>;

struct GoodLocal
{
  void EvaluateResidual(const LocalResidualRequest<GoodSpec>&,
                        LocalResidualResult<GoodSpec>&)
  {}
  void EvaluateJacobian(const LocalJacobianRequest<GoodSpec>&,
                        LocalJacobianResult<GoodSpec>&)
  {}
};
struct GoodCondensation
{
  void Condense(const CondenseRequest<GoodSpec>&, CondenseResult<GoodSpec>&) {}
  void ApplySchur(const ApplySchurRequest<GoodSpec>&,
                  ApplySchurResult<GoodSpec>&)
  {}
};
struct GoodAssembly
{
  void Assemble(const TraceAssemblyRequest<GoodSpec>&,
                TraceAssemblyResult<GoodSpec>&)
  {}
};
struct GoodSolver
{
  void Solve(const TraceSolveRequest<GoodSpec>&, TraceSolveResult<GoodSpec>&) {}
};
struct GoodRecovery
{
  void Recover(const RecoveryRequest<GoodSpec>&, RecoveryResult<GoodSpec>&) {}
};

} // namespace hdg_negative

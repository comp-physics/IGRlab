#pragma once

/**
 * @file sim/hdg/main_backend.hpp
 * @brief Opt-in profile-selected main-application adapter for generic Euler HDG.
 */

#include <functional>
#include <memory>

#include <mpi.h>

#include "mfem.hpp"

#include "sim/config/case_config.hpp"
#include "sim/context/case_context.hpp"
#include "sim/wiring/euler_boundary_plan.hpp"

namespace dg_sim::hdg
{

/**
 * Physics data for one generic Euler HDG run.
 *
 * The case factory owns its analytic coefficients.  The main adapter borrows
 * each coefficient only while projecting it onto its caller-owned HDG volume
 * fields, and retains the resulting semantic boundary plan only long enough
 * to lower it.  A missing static_source denotes the exact zero source.
 */
struct HdgEulerCaseData
{
  std::unique_ptr<mfem::VectorCoefficient> initial_condition;
  std::unique_ptr<mfem::VectorCoefficient> static_source;
  EulerBoundaryPlan boundary_plan;
};

/**
 * Case-owned physics factory invoked after the generic adapter has resolved
 * and constructed its backend-neutral mesh context.
 */
using HdgEulerCaseDataFactory =
  std::function<HdgEulerCaseData(const CaseContext&)>;

/** Borrowed inputs for one main-application HDG run. */
struct HdgEulerMainRequest
{
  const CaseConfig& case_config;
  const CommonOptions& options;
  const HdgEulerCaseDataFactory& make_case_data;
  MPI_Comm communicator = MPI_COMM_WORLD;
};

/**
 * Run the profile-selected generic conventional-Euler HDG backend to
 * completion.
 *
 * The CPU profile and the CUDA profile provide distinct definitions of this
 * symbol and are never linked together.  The CUDA definition is deliberately
 * narrower: it admits only the fixed periodic W8A borrowed-data envelope.
 * Generic restart, CFL, IMEX, and GLVis remain outside both main adapters.
 * This is a collective call: every member of @p request.communicator must
 * enter it with a live, congruent communicator and rank-consistent resolved
 * configuration and mesh content.  It invokes request.make_case_data only
 * after the collective mesh construction converges; each rank must then
 * produce equivalent semantic boundary data and finite four-component fields.
 */
int RunEulerMainBackend(const HdgEulerMainRequest& request);

} // namespace dg_sim::hdg

#pragma once

/**
 * @file sim/hdg/main_backend.hpp
 * @brief Opt-in profile-selected main-application adapter for generic Euler HDG.
 */

#include <mpi.h>

#include "mfem.hpp"

#include "sim/config/case_config.hpp"
#include "sim/wiring/euler_boundary_plan.hpp"

namespace dg_sim::hdg
{

/** Borrowed inputs for one main-application HDG run. */
struct HdgEulerMainRequest
{
  const CaseConfig& case_config;
  const CommonOptions& options;
  mfem::VectorCoefficient& initial_condition;
  const EulerBoundaryPlan& boundary_plan;
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
 * configuration, boundary data, and mesh content.
 */
int RunEulerMainBackend(const HdgEulerMainRequest& request);

} // namespace dg_sim::hdg

#pragma once

/**
 * @file sim/hdg/main_backend.hpp
 * @brief Opt-in main-application adapter for generic Euler HDG on CPU.
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
 * Run the generic conventional-Euler HDG CPU backend to completion.
 *
 * V1 supports fixed Backward Euler, caller-selected output cadence, p1-p3,
 * and MFEM-owned serial/MPI mesh partitioning.  Generic restart, CFL, IMEX,
 * GLVis, and CUDA execution are intentionally outside this first slice.
 * This is a collective call: every member of @p request.communicator must
 * enter it with a live, congruent communicator and rank-consistent resolved
 * configuration, boundary data, and mesh content.
 */
int RunEulerMainBackend(const HdgEulerMainRequest& request);

} // namespace dg_sim::hdg

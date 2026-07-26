#pragma once

/**
 * @file numerics/hdg/euler/smooth_euler_mpi.hpp
 * @brief Conventional CPU/MPI transaction for the smooth-Euler HDG experiment.
 *
 * This default-off, verified-experimental path is deliberately limited to
 * one- and two-rank CPU execution on the frozen two-dimensional periodic
 * manufactured problem.  MFEM owns trace true-DOF sharing and ghost refresh;
 * no application-defined ownership protocol is introduced here.
 */

#include "numerics/hdg/euler/smooth_euler.hpp"
#include "numerics/hdg/trace/trace_dg_mpi.hpp"

#include <mfem.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dg::hdg::euler
{

enum class SmoothEulerMPIStatus : unsigned char
{
  Uninitialized,
  Ready,
  Success
};

/** Test-only fault locations; a selected rank fails before the named phase. */
enum class SmoothEulerMPIFailurePoint : unsigned char
{
  None,
  Assembly,
  Condensation,
  TrueAssembly,
  TraceSolve,
  Distribution,
  Recovery,
  CandidateValidation,
  Commit
};

struct SmoothEulerMPIFailureInjection
{
  SmoothEulerMPIFailurePoint point = SmoothEulerMPIFailurePoint::None;
  int rank = -1;
};

/** Rank-local physical residual/Jacobian and authoritative signed trace map. */
struct SmoothEulerMPILinearization
{
  int local_elements = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  std::vector<double> residual_interior;
  std::vector<double> residual_trace;
  std::vector<double> a, b, c, d;
  std::vector<int> signed_trace_dofs;
};

/** Rank-local Schur blocks and the signed-assembled local trace right side. */
struct SmoothEulerMPICondensation
{
  std::vector<double> schur;
  std::vector<double> right_hand_side;
  std::vector<double> local_assembled_right_hand_side;
};

/**
 * Complete evidence published only by a communicator-wide successful commit.
 * True vectors store this rank's MFEM-owned true-DOF partition.
 */
struct SmoothEulerMPIEvidence
{
  SmoothEulerMPIStatus status = SmoothEulerMPIStatus::Uninitialized;
  int rank = -1;
  int ranks = 0;
  int polynomial_order = 0;
  int elements_per_direction = 0;
  int local_elements = 0;
  int global_elements = 0;
  int local_trace_dofs = 0;
  int local_true_trace_dofs = 0;
  std::int64_t global_true_trace_dofs = 0;
  std::size_t owned_trace_dofs = 0;
  std::size_t ghost_trace_dofs = 0;
  std::size_t global_ghost_trace_dofs = 0;
  int shared_faces = 0;
  int global_shared_faces = 0;
  int negative_signed_trace_dofs = 0;
  int global_negative_signed_trace_dofs = 0;
  int gmres_iterations = 0;
  double gmres_reported_relative_residual = 0.0;
  double true_relative_residual = 0.0;
  double damping = 0.0;
  double initial_nonlinear_residual = 0.0;
  double candidate_nonlinear_residual = 0.0;
  double candidate_minimum_density = 0.0;
  double candidate_minimum_pressure = 0.0;
  std::uint64_t entry_generation = 0;
  std::uint64_t committed_generation = 0;

  SmoothEulerMPILinearization linearization;
  SmoothEulerMPICondensation condensation;
  std::vector<double> entry_interior;
  std::vector<double> entry_trace;
  std::vector<double> final_interior;
  std::vector<double> final_trace;
  std::vector<double> true_right_hand_side;
  std::vector<double> true_trace_increment;
  std::vector<double> distributed_trace_increment;
  std::vector<double> recovered_interior_increment;
  std::vector<double> true_probe;
  std::vector<double> true_probe_action;
  /** Column-major spanning-basis actions; each column stores owned true rows. */
  std::vector<double> true_basis_actions;
  bool passed = false;
};

/**
 * Persistent one-/two-rank transaction owner.
 *
 * Initialize() atomically publishes the periodic MFEM graph and the exact old
 * and perturbed current stage. Execute() stages local assembly, condensation,
 * P^T S_L P/P^T b_L, HYPRE solve, Distribute, recovery, and a damped
 * admissible state. A failed call preserves all published bytes, evidence,
 * status, and generation on every rank.
 */
class SmoothEulerMPIExperiment
{
public:
  SmoothEulerMPIExperiment(MPI_Comm communicator, int polynomial_order);
  ~SmoothEulerMPIExperiment();
  SmoothEulerMPIExperiment(SmoothEulerMPIExperiment&&) noexcept;
  SmoothEulerMPIExperiment& operator=(SmoothEulerMPIExperiment&&) noexcept;
  SmoothEulerMPIExperiment(const SmoothEulerMPIExperiment&) = delete;
  SmoothEulerMPIExperiment& operator=(const SmoothEulerMPIExperiment&) = delete;

  bool Initialize(int elements_per_direction);
  bool Execute(const SmoothEulerMPIFailureInjection& injection = {});

  SmoothEulerMPIStatus status() const noexcept;
  SmoothEulerMPIFailurePoint last_failure_point_reached() const noexcept;
  std::uint64_t generation() const noexcept;
  int polynomial_order() const noexcept { return polynomial_order_; }
  int elements_per_direction() const noexcept;
  const SmoothEulerMPIEvidence& evidence() const noexcept;
  const mfem::ParMesh* mesh() const noexcept;
  const mfem::ParFiniteElementSpace* interior_space() const noexcept;
  const mfem::ParFiniteElementSpace* trace_space() const noexcept;
  const mfem::ParGridFunction* old_interior() const noexcept;
  const mfem::ParGridFunction* current_interior() const noexcept;
  const mfem::ParGridFunction* current_trace() const noexcept;
  const trace::DGInterfaceTraceMap* trace_map() const noexcept;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
  int polynomial_order_ = 0;
  int quadrature_order_ = 0;
};

} // namespace dg::hdg::euler

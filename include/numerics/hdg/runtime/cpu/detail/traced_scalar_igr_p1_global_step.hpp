#pragma once

/**
 * @file numerics/hdg/runtime/cpu/detail/traced_scalar_igr_p1_global_step.hpp
 * @brief Private one-linearization traced-scalar-IGR p1 HDG solve.
 *
 * This helper deliberately has no solver/session, request, coordinate, or
 * publication authority.  Its caller already owns communicator admission,
 * affine-geometry validation, state lifetime, and nonlinear acceptance.
 */

#include "numerics/hdg/runtime/cpu/borrowed_p1_dg_interface_layout.hpp"
#include "numerics/hdg/runtime/cpu/traced_scalar_igr_p1_local.hpp"

#include <mfem.hpp>

namespace dg::hdg::runtime::cpu::detail
{

/** Private assembled-trace solve/recovery seam shared by Compute and tests. */
struct TracedScalarIgrP1TraceSolveInput final
{
  MPI_Comm communicator = MPI_COMM_NULL;
  const BorrowedP1DGInterfaceLayout* layout = nullptr;
  const mfem::ParFiniteElementSpace* trace_space = nullptr;
  const mfem::HypreParMatrix* true_matrix = nullptr;
  const mfem::Vector* true_rhs_values = nullptr;
  const core::HostReferenceEngine* condensation = nullptr;
};

struct TracedScalarIgrP1TraceSolveOutput final
{
  mfem::Vector trace_increment;
  mfem::Vector interior_increment;
  double reduced_rhs_norm = 0.0;
  double linear_residual_norm = 0.0;
};

/**
 * Solve the assembled true trace system and recover local interior increments.
 * An exactly zero true RHS skips GMRES but still distributes and recovers.  On
 * failure output is unchanged.
 */
bool SolveAndRecoverTracedScalarIgrP1Trace(
    const TracedScalarIgrP1TraceSolveInput& input,
    TracedScalarIgrP1TraceSolveOutput& output) noexcept;

struct TracedScalarIgrP1GlobalStepInput final
{
  MPI_Comm communicator = MPI_COMM_NULL;
  const BorrowedP1DGInterfaceLayout* layout = nullptr;

  /** One caller-validated affine quadrilateral per local element. */
  const traced_scalar_igr_p1::AffineQuad* affine_quads = nullptr;
  int affine_quad_count = 0;
  traced_scalar_igr_p1::Parameters parameters{};

  /** The iterate and fixed preceding-time state remain distinct inputs. */
  const mfem::Vector* current_true_trace = nullptr;
  const mfem::Vector* previous_true_trace = nullptr;
  const mfem::Vector* current_interior = nullptr;
  const mfem::Vector* previous_interior = nullptr;

  /**
   * Optional local four-component p1 L2 source.  It must use byNODES
   * ordering; an absent or empty coefficient vector denotes the zero source.
   * The helper interpolates it at every local volume quadrature point.
   */
  const mfem::ParFiniteElementSpace* static_euler_rhs_space = nullptr;
  const mfem::Vector* static_euler_rhs_coefficients = nullptr;
};

struct TracedScalarIgrP1GlobalStepOutput final
{
  mfem::Vector trace_increment;
  mfem::Vector interior_increment;
  double initial_residual_norm = 0.0;
  double reduced_rhs_norm = 0.0;
  double linear_residual_norm = 0.0;
};

/**
 * Compute exactly one global Newton linearization.  On false, @a output is
 * unchanged.  All ranks must call this collectively with compatible input.
 */
bool ComputeTracedScalarIgrP1GlobalStep(
    const TracedScalarIgrP1GlobalStepInput& input,
    TracedScalarIgrP1GlobalStepOutput& output) noexcept;

} // namespace dg::hdg::runtime::cpu::detail

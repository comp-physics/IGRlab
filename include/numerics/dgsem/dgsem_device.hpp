#pragma once

/**
 * @file numerics/dgsem/dgsem_device.hpp
 * @brief Correctness-first MFEM device kernels for the frozen DGSEM L2 scope.
 * @ingroup numerics_dgsem
 *
 * These functions are deliberately data-only. Mesh traversal, element/face
 * discovery, orientation lookup, boundary-state construction, and MPI exchange
 * remain host-once setup. The packed kernels preserve MFEM's component-major
 * element layout: entry `(row, component)` is stored as
 * `row + matrix.Height()*component`.
 */

#include "mfem.hpp"


namespace mfem
{
namespace dgsem
{

/// Ownership policy for a packed face whose normal points out of side one.
enum class DGSEMDeviceFaceOwnership : int
{
  Local = 0,   ///< Accumulate equal-and-opposite contributions on both sides.
  Shared = 1,  ///< Accumulate only the locally owned (left) contribution.
  Boundary = 2 ///< Accumulate only the interior (left) contribution.
};

/// Host-visible snapshot of the MFEM runtime selected before kernel launches.
struct DGSEMDeviceRuntimeInfo
{
  bool configured = false;
  bool enabled = false;
  bool cuda = false;
  int device_id = 0;
  int device_count = 0;
  MemoryType host_memory_type = MemoryType::HOST;
  MemoryType device_memory_type = MemoryType::HOST;
  MemoryClass host_memory_class = MemoryClass::HOST;
  MemoryClass device_memory_class = MemoryClass::HOST;
};

DGSEMDeviceRuntimeInfo GetDGSEMDeviceRuntimeInfo();

/// Fail-closed status written by the packed Euler+IGR admission scan.
enum class DGSEMDeviceStateStatus : int
{
  Valid = 0,
  InvalidSigma = 1,
  InvalidDensity = 2,
  DensityBelowFloor = 3,
  InvalidPressure = 4
};

/// Fixed storage for the deterministic packed-state admission reduction.
constexpr int kDGSEMDeviceStateScanWorkspaceSize = 256 * 7;

/** Validate packed element/face panels and reduce frozen-sigma telemetry.
 *
 * Exactly one of `scan_owned_states` and `scan_face_states` is true. `slots`
 * is a persistent fixed-lane reduction workspace; `result` is a persistent
 * six-double workspace containing status, sample count, nonpositive count,
 * nonfinite count, minimum q_sigma, and minimum q_sigma/(gamma*p). The scan
 * executes where the input panels reside; callers need transfer only these
 * bounded reduction scalars.
 */
void ScanEulerIGRStatePanelsDevice(int dim,
                                   real_t gamma,
                                   real_t density_floor,
                                   bool scan_owned_states,
                                   bool scan_face_states,
                                   const DenseMatrix& states,
                                   const Vector& sigma,
                                   const DenseMatrix& left_states,
                                   const Vector& left_sigma,
                                   const DenseMatrix& right_states,
                                   const Vector& right_sigma,
                                   Array<double>& slots,
                                   Array<double>& result);

/** Gather a local MFEM L-vector into packed element/component columns.
 *
 * `element_vdofs` is ordered `(element,equation,local)` with local fastest.
 * Signed MFEM vdofs are interpreted with the standard `-1-v` convention.
 */
void GatherElementStateDevice(int ne,
                              int dof,
                              int num_equations,
                              const Array<int>& element_vdofs,
                              const Vector& local_state,
                              DenseMatrix& packed_state);

/** Interpolate a scalar finite-element field to packed DGSEM state nodes. */
void InterpolateElementScalarDevice(int rows,
                                    int scalar_dofs_per_element,
                                    const Array<int>& scalar_element_dofs,
                                    const DenseMatrix& interpolation,
                                    const Vector& scalar_field,
                                    Vector& packed_scalar);

/** Interpolate shared-neighbor scalar data to packed face points.
 *
 * The interpolation and signed neighbor map have height `nfaces`; rows for
 * non-shared faces may be all zero.
 */
void InterpolateSharedFaceScalarDevice(
  int nfaces,
  int scalar_dofs_per_element,
  const Array<int>& scalar_face_neighbor_dofs,
  const DenseMatrix& interpolation,
  const Vector& scalar_face_neighbor_data,
  Vector& packed_shared_scalar);

/** Gather persistent nodal face maps and apply lowered Euler boundaries.
 *
 * Local and shared rows obtain their right state from the local packed batch
 * or MFEM face-neighbor vector, respectively. Boundary rows consume a packed
 * boundary kind and conservative reference state prepared during host-once
 * topology setup. The point kernel has no runtime polymorphism.
 */
void GatherEulerRusanovFacesDevice(
  int dim,
  int ne,
  int dof,
  const DenseMatrix& packed_states,
  const Vector& packed_sigma,
  const Vector& face_neighbor_state,
  const Array<int>& shared_state_vdofs,
  const Vector& shared_sigma,
  const DenseMatrix& normals,
  const Array<int>& left_elements,
  const Array<int>& left_local_dofs,
  const Array<int>& right_elements,
  const Array<int>& right_local_dofs,
  const Array<int>& ownership,
  const Array<int>& boundary_kinds,
  const DenseMatrix& boundary_reference_states,
  const Vector& boundary_gammas,
  const Vector& boundary_outlet_pressures,
  const Vector& boundary_density_floors,
  const Vector& boundary_pressure_floors,
  DenseMatrix& left_states,
  DenseMatrix& right_states,
  Vector& left_sigma,
  Vector& right_sigma);

/// Add two identically shaped packed matrices without host staging.
void AddPackedMatricesDevice(const DenseMatrix& addend,
                             DenseMatrix& inout);

/** Scatter a packed element/component matrix to a local MFEM L-vector.
 *
 * The metadata preflight guarantees a one-to-one DG map, so this kernel does
 * not require atomics and overwrites every local entry exactly once.
 */
void ScatterElementStateDevice(int ne,
                               int dof,
                               int num_equations,
                               const Array<int>& element_vdofs,
                               const DenseMatrix& packed_state,
                               Vector& local_state);

/** Evaluate the frozen LGL moment, derivative, SBP, and split-form identities.
 *
 * `derivative(degree*n+i)` is represented as a DenseMatrix with height `n`;
 * the other matrices use normal MFEM column-major storage.
 */
void EvaluateLGLBasisDevice(const Vector& nodes,
                            const Vector& weights,
                            const DenseMatrix& derivative_matrix,
                            const DenseMatrix& split_matrix,
                            int order,
                            Vector& moments,
                            DenseMatrix& derivative_action,
                            DenseMatrix& sbp_residual,
                            DenseMatrix& split_residual);

/** Build 2D curl-form metrics from packed nodal coordinates.
 *
 * `coordinates` has height `ne*dof` and two columns. `lex_to_local` is the
 * element-local tensor map shared by the uniform batch. `metrics` has four
 * columns ordered `(reference,physical) = (0,0),(0,1),(1,0),(1,1)`.
 */
void BuildContravariantMetrics2DDevice(int ne,
                                       int n1d,
                                       const Array<int>& lex_to_local,
                                       const DenseMatrix& derivative_matrix,
                                       const DenseMatrix& coordinates,
                                       DenseMatrix& metrics);

/** Build Euler plus stage-frozen scalar-sigma contravariant nodal fluxes.
 *
 * The pressure-like sigma correction exactly matches FluxWithIGR's accepted
 * tensor semantics. It does not update sigma. When
 * `frozen_sigma_spectral_speed` is true, nodal CFL speeds use the exact local
 * frozen-sigma acoustic radius on the independently admitted q_sigma>0 set.
 * `metrics` has `dim*dim` columns ordered
 * `reference*dim + physical`. The result has `dim*num_equations` columns
 * ordered `reference*num_equations + equation`.
 */
void BuildEulerContravariantFluxDevice(int dim,
                                       real_t gamma,
                                       real_t velocity_density_floor,
                                       const DenseMatrix& states,
                                       const Vector& sigma,
                                       const DenseMatrix& metrics,
                                       DenseMatrix& contravariant_flux,
                                       Vector& nodal_speeds,
                                       bool frozen_sigma_spectral_speed = false);

/** Evaluate only the packed nodal Euler characteristic-speed bound.
 *
 * This is the speed half of BuildEulerContravariantFluxDevice() and shares
 * its exact ordinary/frozen-sigma formula. It does not build fluxes or write a
 * residual, so an accepted endpoint can refresh the next-step CFL state
 * without repeating the transport operator.
 */
void ComputeEulerNodalCharacteristicSpeedsDevice(
  int dim,
  real_t gamma,
  real_t velocity_density_floor,
  const DenseMatrix& states,
  const Vector& sigma,
  Vector& nodal_speeds,
  bool frozen_sigma_spectral_speed = false);

/** Apply the accepted central symmetric DGSEM volume contraction.
 *
 * All elements in the batch have the same `n1d`, dimension, and local tensor
 * map. The output is the quadrature-weighted weak volume contribution.
 */
void ApplyCentralVolumeDevice(int dim,
                              int n1d,
                              const Array<int>& lex_to_local,
                              const DenseMatrix& split_matrix,
                              const Vector& weights,
                              const DenseMatrix& contravariant_flux,
                              DenseMatrix& weak_volume);

/// Number of unique coordinate-line pairs used by the LLF volume kernel.
int DGSEMVolumePairCount(int dim, int ne, int n1d);

/** Add conservative local-Lax--Friedrichs volume dissipation on device.
 *
 * A first kernel evaluates each unique unordered nodal pair once and stores
 * all Euler components.  A second deterministic gather applies Dsplit in the
 * exact legacy pair orientation/order without atomics.  `pair_dissipation`
 * is a persistent `(DGSEMVolumePairCount x num_equations)` workspace. With
 * frozen IGR sigma, the pair scale uses
 * `|u.g| + sqrt(|gamma*p + (gamma-1)*sigma|/rho)*|g|`. For positive q_sigma
 * this is the exact local acoustic spectral-radius bound. The modulus branch
 * at nonpositive q_sigma is finite only so a rejected stage can unwind; the
 * independent FP64 stage-admission scan remains the hyperbolicity gate.
 */
void AddEulerLLFVolumeDissipationDevice(
  int dim,
  int ne,
  int n1d,
  real_t gamma,
  real_t velocity_density_floor,
  const Array<int>& lex_to_local,
  const DenseMatrix& split_matrix,
  const Vector& weights,
  const DenseMatrix& states,
  const Vector& sigma,
  const DenseMatrix& metrics,
  DenseMatrix& pair_dissipation,
  DenseMatrix& weak_volume);

/** Assemble packed Euler Rusanov face contributions deterministically.
 *
 * Face states and normals are prepared host-once after MFEM orientation and
 * exchange setup. Host-once CSR adjacency lists the incident face ids and
 * deterministic signs for each element row. Surface accumulation therefore
 * remains atomic-free and face-order deterministic without scanning every
 * packed face for every output row.
 */
void AssembleEulerRusanovFacesDevice(int dim,
                                     int ne,
                                     int dof,
                                     real_t gamma,
                                     real_t velocity_density_floor,
                                     const DenseMatrix& left_states,
                                     const DenseMatrix& right_states,
                                     const DenseMatrix& normals,
                                     const Vector& left_sigma,
                                     const Vector& right_sigma,
                                     const Vector& face_weights,
                                     const Array<int>& row_face_offsets,
                                     const Array<int>& row_face_ids,
                                     const Array<int>& row_face_signs,
                                     DenseMatrix& face_flux,
                                     Vector& face_speeds,
                                     DenseMatrix& weak_surface,
                                     bool frozen_sigma_spectral_speed = false);

/** Evaluate only the packed two-sided Rusanov characteristic-speed bound.
 *
 * The input panels must already contain the accepted local, shared, and
 * physical-boundary ghost states. This shares the exact speed formula used by
 * AssembleEulerRusanovFacesDevice() and writes neither flux nor residual.
 */
void ComputeEulerRusanovFaceCharacteristicSpeedsDevice(
  int dim,
  real_t gamma,
  real_t velocity_density_floor,
  const DenseMatrix& left_states,
  const DenseMatrix& right_states,
  const DenseMatrix& normals,
  const Vector& left_sigma,
  const Vector& right_sigma,
  Vector& face_speeds,
  bool frozen_sigma_spectral_speed = false);

/** Integrate the outward numerical flux over physical boundary faces.
 *
 * `face_flux` is the unweighted flux produced by
 * AssembleEulerRusanovFacesDevice(). Only packed entries whose ownership is
 * `Boundary` contribute. The deterministic component-wise reduction uses the
 * same face weights and scaled outward normals as the residual, and overwrites
 * `boundary_flux` with one rank-local value per equation.
 */
void IntegratePhysicalBoundaryFluxDevice(const Array<int>& ownership,
                                         const Vector& face_weights,
                                         const DenseMatrix& face_flux,
                                         Vector& boundary_flux);

/** Apply a packed batch of element inverse-mass matrices.
 *
 * `inverse_mass` has height `ne*dof`. Width `dof` stores row `i` of element
 * `e`'s inverse matrix at row `e*dof+i`; width one stores only its diagonal.
 * The compact representation is an exact algebraic optimization: callers
 * must verify that every omitted off-diagonal entry is exactly zero.
 */
void ApplyInverseMassDevice(int ne,
                            int dof,
                            const DenseMatrix& inverse_mass,
                            const DenseMatrix& weak,
                            DenseMatrix& strong);

/// Deterministic component sums, one MFEM kernel iteration per component.
void ComponentSumsDevice(const DenseMatrix& values, Vector& sums);

/// Deterministic nonnegative maximum, evaluated by one MFEM kernel iteration.
void DeterministicMaximumDevice(const Vector& values, Vector& maximum);

} // namespace dgsem
} // namespace mfem

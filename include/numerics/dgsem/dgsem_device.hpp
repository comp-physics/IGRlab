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
 * tensor semantics. It does not update sigma and does not enable sigma-aware
 * CFL or Rusanov speeds. `metrics` has `dim*dim` columns ordered
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
                                       Vector& nodal_speeds);

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

/** Assemble packed Euler Rusanov face contributions deterministically.
 *
 * Face states and normals are prepared host-once after MFEM orientation and
 * exchange setup. A local face writes both elements, while shared and boundary
 * faces write only the locally owned left element. Each output entry scans the
 * face list in its declared order, avoiding atomics and preserving a stable
 * correctness reference.
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
                                     const Array<int>& left_elements,
                                     const Array<int>& left_local_dofs,
                                     const Array<int>& right_elements,
                                     const Array<int>& right_local_dofs,
                                     const Array<int>& ownership,
                                     DenseMatrix& face_flux,
                                     Vector& face_speeds,
                                     DenseMatrix& weak_surface);

/** Apply a packed batch of element inverse-mass matrices.
 *
 * `inverse_mass` has height `ne*dof` and width `dof`; row `e*dof+i` contains
 * row `i` of element `e`'s inverse matrix.
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

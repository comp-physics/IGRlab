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

#include <cstddef>
#include <memory>

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
  real_t gamma,
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
  const Vector& boundary_density_floors,
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

/** Build the same nodal flux directly in the wide volume-panel layout.
 *
 * The panel has shape `(dim*dof) x (ne*num_equations)`.  Its column
 * `element + ne*equation` contains all reference-direction fluxes for one
 * element/equation pair.  This layout feeds the shared reference operator
 * without a packing pass.
 */
void BuildEulerContravariantFluxPanelDevice(
  int dim,
  int ne,
  int dof,
  real_t gamma,
  real_t velocity_density_floor,
  const DenseMatrix& states,
  const Vector& sigma,
  const DenseMatrix& metrics,
  DenseMatrix& contravariant_flux_panel,
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

/** Build the shared reference-only central-volume panel operator.
 *
 * Metric and state dependence remain in the contravariant flux panel, so one
 * `dof x (dim*dof)` operator is shared by every element and Euler equation,
 * including on bilinear quadrilateral geometry.
 */
void BuildCentralVolumePanelOperator(int dim,
                                     int n1d,
                                     const Array<int>& lex_to_local,
                                     const DenseMatrix& split_matrix,
                                     const Vector& weights,
                                     DenseTensor& panel_operator);

/** Apply the shared central-volume operator with native FP64 GPU BLAS.
 *
 * The input/output matrices must use the layouts documented by
 * BuildEulerContravariantFluxPanelDevice().  This entry point is deliberately
 * precision-specific: the separately selected fixed-point emulation screen is
 * certified against this path rather than silently changing its arithmetic.
 */
void ApplyCentralVolumePanelFP64Device(
  int dim,
  int ne,
  int dof,
  int num_equations,
  const DenseTensor& panel_operator,
  const DenseMatrix& contravariant_flux_panel,
  DenseMatrix& weak_volume);

/** Result of one opt-in cuBLAS fixed-point FP64-emulation operation.
 *
 * The emulated operation keeps FP64 inputs and outputs.  cuBLAS performs
 * row/column power-of-two scaling and an Ozaki fixed-point decomposition
 * internally; this status makes both the selected precision and any native
 * FP64 fallback observable to the caller.
 */
struct DGSEMFixedPointPanelStatus
{
  bool compiled = false;
  bool initialized = false;
  bool launch_succeeded = false;
  bool emulation_observed = false;
  bool retained_status_valid = false;
  int cublas_status = 0;
  int cuda_status = 0;
  int retained_mantissa_bits = -1;
  int max_mantissa_bits = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  int m = 0;
  int n = 0;
  int k = 0;
  std::size_t workspace_bytes = 0;
};

/** Persistent, dedicated cuBLAS context for Ozaki fixed-point FP64 GEMM.
 *
 * A dedicated handle prevents the experimental emulation controls from
 * mutating MFEM's process-global GPU BLAS handle.  Initialization allocates
 * all workspace once; Mult() performs no allocation or synchronization.
 */
class DGSEMFixedPointVolumeContext
{
public:
  DGSEMFixedPointVolumeContext();
  ~DGSEMFixedPointVolumeContext();
  DGSEMFixedPointVolumeContext(const DGSEMFixedPointVolumeContext&) = delete;
  DGSEMFixedPointVolumeContext& operator=(
    const DGSEMFixedPointVolumeContext&) = delete;

  DGSEMFixedPointPanelStatus Initialize(int m,
                                        int n,
                                        int k,
                                        int max_mantissa_bits,
                                        std::size_t workspace_limit_bytes);

  DGSEMFixedPointPanelStatus Mult(
    const DenseTensor& panel_operator,
    const DenseMatrix& contravariant_flux_panel,
    DenseMatrix& weak_volume);

  /// Synchronize only the retained-bit telemetry after an audited operation.
  DGSEMFixedPointPanelStatus SynchronizeStatus();

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

/** Reduce an emulated/native panel comparison to three FP64 audit scalars.
 *
 * The output is `(max_abs_reference, max_abs_error, non_finite_count)`.
 * This is deliberately independent of the emulated cuBLAS handle.
 */
void AuditCentralVolumePanelDevice(const DenseMatrix& candidate,
                                   const DenseMatrix& reference,
                                   Vector& audit);

/// Copy identically shaped packed matrices without host staging.
void CopyPackedMatrixDevice(const DenseMatrix& source, DenseMatrix& target);

/// Number of unique coordinate-line pairs used by the LLF volume kernel.
int DGSEMVolumePairCount(int dim, int ne, int n1d);

/** Add conservative local-Lax--Friedrichs volume dissipation on device.
 *
 * A first kernel evaluates each unique unordered nodal pair once and stores
 * all Euler components.  A second deterministic gather applies Dsplit in the
 * exact legacy pair orientation/order without atomics.  `pair_dissipation`
 * is a persistent `(DGSEMVolumePairCount x num_equations)` workspace.
 */
void AddEulerLLFVolumeDissipationDevice(
  int dim,
  int ne,
  int n1d,
  real_t gamma,
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

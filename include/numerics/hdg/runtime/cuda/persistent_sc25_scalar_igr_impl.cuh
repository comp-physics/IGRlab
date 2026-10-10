#pragma once

#include "numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr.cuh"
#include "numerics/hdg/runtime/cuda/traced_scalar_igr_p1_private.cuh"

#include <cstddef>
#include <cstdint>

namespace dg::hdg::runtime::cuda_w8b::sc25::detail
{

enum class KernelOperation : int
{
  initialize_projection,
  initialize_fixture,
  clear_assembly,
  assemble,
  auxiliary_identity_rows,
  condense_and_scatter,
  recover_direction,
  apply_trial,
  accept_trial,
  pack_checkpoint,
  unpack_checkpoint,
  reduce_state,
  hash_state,
  hash_topology,
  copy_slot
};

/**
 * Fixed pointer-only view into the two Umpire arenas and persistent HYPRE
 * buffers.  It is trivially copyable and passed by value to the three leaf
 * kernels.  The private interior is deliberately field-blocked:
 *
 *   U[e][c][a]       : [0, 1024)
 *   Sigma[e][a]      : [1024, 1280)
 *   z[e][c][a]       : [1280, 1792)
 *
 * It is never represented as seven component slabs inside a state slot.
 */
struct DeviceView final
{
  int elements = 0;
  int scalar_true_trace_nodes = 0;
  int full_unknowns = 0;
  int true_trace_unknowns = 0;
  int volume_quadrature_points = 0;
  int face_quadrature_points = 0;
  int true_matrix_nonzeros = 0;
  int assemble_dense_jacobian = 0;
  int auxiliary_identity_rows = 0;
  int operation = 0;
  int committed_slot = 0;
  int selected_current_slot = 0;
  int compiled_architecture_expected = required_native_architecture;
  double dt = fixed_dt;
  double trial_theta = 0.0;

  double* slot_interior[2]{};
  double* slot_trace[2]{};
  double* trial_interior = nullptr;
  double* trial_trace = nullptr;
  double* direction_interior = nullptr;
  double* direction_trace = nullptr;

  double* a = nullptr;
  double* b = nullptr;
  double* c = nullptr;
  double* d = nullptr;
  double* local_residual = nullptr;
  double* trace_residual = nullptr;
  double* lu = nullptr;
  double* inverse_residual = nullptr;
  double* inverse_b = nullptr;
  double* schur = nullptr;
  double* condensed_rhs = nullptr;
  double* recovery_rhs = nullptr;
  double* recovered = nullptr;
  double* full_residual = nullptr;
  double* full_jacobian = nullptr;
  double* reduction = nullptr;
  // Fixed adapter storage for the borrowed packed P1 algebra.  These slices
  // are carved from the owner arena before its allocation baseline freezes.
  double* packed_vertices = nullptr;
  common::traced_scalar_igr_p1_local_algebra::Parameters* packed_parameters =
      nullptr;
  double* packed_interior = nullptr;
  double* packed_previous_interior = nullptr;
  double* packed_trace = nullptr;
  double* packed_previous_trace = nullptr;
  double* packed_f = nullptr;
  double* packed_g = nullptr;
  std::uint8_t* packed_status = nullptr;
  double* packed_volume_xyw = nullptr;
  double* packed_segment_xw = nullptr;
  double* checkpoint_interior_slabs = nullptr;
  double* checkpoint_trace_slabs = nullptr;
  std::uint8_t* state_hash_bytes = nullptr;

  double* volume_weight = nullptr;
  double* volume_shape = nullptr;
  double* volume_derivative = nullptr;
  double* volume_x = nullptr;
  double* volume_y = nullptr;
  double* face_weight = nullptr;
  double* face_height = nullptr;
  double* face_interior_shape = nullptr;
  double* face_oriented_shape = nullptr;
  double* face_normal = nullptr;
  double* face_x = nullptr;
  double* face_y = nullptr;
  double* volume_node_x = nullptr;
  double* volume_node_y = nullptr;
  double* trace_node_coordinate = nullptr;

  double* true_matrix_data = nullptr;
  double* true_rhs = nullptr;
  const double* true_solution = nullptr;

  int* local_trace_to_true = nullptr;
  int* local_trace_sign = nullptr;
  int* element_to_true_csr = nullptr;
  int* true_matrix_row_offsets = nullptr;
  int* true_matrix_columns = nullptr;
  int* row_pivots = nullptr;
  int* column_permutation = nullptr;
  int* status = nullptr;
  int* compiled_architecture = nullptr;
};

static_assert(std::is_trivially_copyable<DeviceView>::value,
              "kernel view must remain a static POD");

enum DeviceStatusBit : int
{
  device_success = 0,
  device_nonfinite = 1 << 0,
  device_admissibility = 1 << 1,
  device_singular_local_block = 1 << 2,
  device_local_application = 1 << 3,
  device_invalid_layout = 1 << 4,
  device_wrong_architecture = 1 << 5
};

} // namespace dg::hdg::runtime::cuda_w8b::sc25::detail

extern "C" __global__ void HdgSc25AssembleLocalFp64Kernel(
    dg::hdg::runtime::cuda_w8b::sc25::detail::DeviceView view);
extern "C" __global__ void HdgSc25CondenseFp64Kernel(
    dg::hdg::runtime::cuda_w8b::sc25::detail::DeviceView view);
extern "C" __global__ void HdgSc25RecoverFp64Kernel(
    dg::hdg::runtime::cuda_w8b::sc25::detail::DeviceView view);

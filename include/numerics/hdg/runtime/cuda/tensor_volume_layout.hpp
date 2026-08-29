#pragma once

/**
 * @file numerics/hdg/runtime/cuda/tensor_volume_layout.hpp
 * @brief Checked compact scratch layout for FP64 tensor volume assembly.
 *
 * This is immutable numerical-layout policy only.  It contains no run events,
 * iteration history, replay data, or observability interface.
 */

#include <cstddef>
#include <limits>

namespace dg::hdg::runtime::cuda_w8a::detail
{

inline constexpr int tensor_volume_conserved_components = 4;
inline constexpr int tensor_volume_igr_interior_components = 7;
inline constexpr int tensor_volume_igr_flux_columns = 5;
inline constexpr int tensor_volume_features = 3;
inline constexpr int tensor_volume_mma_inner = 4;

inline constexpr int tensor_volume_temporal_offset = 0;
inline constexpr int tensor_volume_flux_x_offset = 4;
inline constexpr int tensor_volume_flux_y_offset = 8;
inline constexpr int tensor_volume_jacobian_x_offset = 12;
inline constexpr int tensor_volume_conventional_stride = 44;
inline constexpr int tensor_volume_igr_common_stride = 52;
inline constexpr int tensor_volume_igr_rho_offset = 52;
inline constexpr int tensor_volume_igr_sigma_offset = 53;
inline constexpr int tensor_volume_igr_z_x_offset = 54;
inline constexpr int tensor_volume_igr_z_y_offset = 55;
inline constexpr int tensor_volume_igr_q_offset = 56;
inline constexpr int tensor_volume_igr_dq_offset = 57;
inline constexpr int tensor_volume_igr_stride = 66;

constexpr int TensorVolumeCoefficientStride(const bool igr) noexcept
{
  return igr ? tensor_volume_igr_stride
             : tensor_volume_conventional_stride;
}

constexpr int TensorVolumeFluxColumns(const bool igr) noexcept
{
  return igr ? tensor_volume_igr_flux_columns
             : tensor_volume_conserved_components;
}

constexpr int TensorVolumeTemporalSlot(const int component) noexcept
{
  return component >= 0 && component < tensor_volume_conserved_components
      ? tensor_volume_temporal_offset + component : -1;
}

constexpr int TensorVolumeFluxXSlot(const int component) noexcept
{
  return component >= 0 && component < tensor_volume_conserved_components
      ? tensor_volume_flux_x_offset + component : -1;
}

constexpr int TensorVolumeFluxYSlot(const int component) noexcept
{
  return component >= 0 && component < tensor_volume_conserved_components
      ? tensor_volume_flux_y_offset + component : -1;
}

constexpr int TensorVolumeJacobianXSlot(
    const bool igr, const int row, const int column) noexcept
{
  const int columns = TensorVolumeFluxColumns(igr);
  return row >= 0 && row < tensor_volume_conserved_components &&
          column >= 0 && column < columns
      ? tensor_volume_jacobian_x_offset + row * columns + column : -1;
}

constexpr int TensorVolumeJacobianYOffset(
    const bool igr) noexcept
{
  return tensor_volume_jacobian_x_offset +
      tensor_volume_conserved_components * TensorVolumeFluxColumns(igr);
}

constexpr int TensorVolumeJacobianYSlot(
    const bool igr, const int row, const int column) noexcept
{
  const int columns = TensorVolumeFluxColumns(igr);
  return row >= 0 && row < tensor_volume_conserved_components &&
          column >= 0 && column < columns
      ? TensorVolumeJacobianYOffset(igr) + row * columns + column : -1;
}

// dQ for trial variable rho, rho*u, or rho*v is represented exactly as
// a_b phi + bx_b d_x(phi) + by_b d_y(phi).  Energy has zero dQ.
constexpr int TensorVolumeIgrDqSlot(
    const int variable, const int feature) noexcept
{
  return variable >= 0 && variable < 3 && feature >= 0 &&
          feature < tensor_volume_features
      ? tensor_volume_igr_dq_offset +
            variable * tensor_volume_features + feature
      : -1;
}

struct TensorVolumeScratchPlan final
{
  std::size_t workspace_doubles = 0;
  int coefficient_stride = 0;
  int capacity_points = 0;
  int chunk_points = 0;
  bool valid = false;
};

/**
 * Plan one element's zero-allocation coefficient stream.
 *
 * A final chunk may have any positive length.  If all quadrature points do
 * not fit, every preceding chunk is rounded down to a positive m8n8k4 inner
 * multiple.  The caller advances by chunk_points until the final remainder.
 */
constexpr TensorVolumeScratchPlan PlanTensorVolumeScratch(
    const int interior_dofs, const int trace_dofs,
    const int volume_points, const bool igr) noexcept
{
  TensorVolumeScratchPlan plan{};
  if (interior_dofs <= 0 || trace_dofs <= 0 || volume_points <= 0)
    return plan;
  const std::size_t interior = static_cast<std::size_t>(interior_dofs);
  const std::size_t trace_plus_one =
      static_cast<std::size_t>(trace_dofs) + 1U;
  if (trace_plus_one == 0 ||
      interior > std::numeric_limits<std::size_t>::max() / trace_plus_one)
    return plan;
  plan.workspace_doubles = interior * trace_plus_one;
  plan.coefficient_stride = TensorVolumeCoefficientStride(igr);
  const std::size_t capacity_points = plan.workspace_doubles /
      static_cast<std::size_t>(plan.coefficient_stride);
  if (capacity_points >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    return TensorVolumeScratchPlan{};
  plan.capacity_points = static_cast<int>(capacity_points);
  if (plan.capacity_points <= 0) return plan;
  if (volume_points <= plan.capacity_points)
  {
    plan.chunk_points = volume_points;
  }
  else
  {
    plan.chunk_points = plan.capacity_points -
        plan.capacity_points % tensor_volume_mma_inner;
    if (plan.chunk_points < tensor_volume_mma_inner) return plan;
  }
  plan.valid = static_cast<std::size_t>(plan.chunk_points) <=
      plan.workspace_doubles /
          static_cast<std::size_t>(plan.coefficient_stride);
  return plan;
}

constexpr bool TensorVolumeCoefficientOffset(
    const TensorVolumeScratchPlan& plan, const int point,
    const int slot, std::size_t& offset) noexcept
{
  if (!plan.valid || point < 0 || point >= plan.chunk_points ||
      slot < 0 || slot >= plan.coefficient_stride)
    return false;
  const std::size_t local_point = static_cast<std::size_t>(point);
  const std::size_t stride =
      static_cast<std::size_t>(plan.coefficient_stride);
  if (local_point > std::numeric_limits<std::size_t>::max() / stride)
    return false;
  offset = local_point * stride + static_cast<std::size_t>(slot);
  return offset < plan.workspace_doubles;
}

static_assert(TensorVolumeJacobianYOffset(false) == 28);
static_assert(TensorVolumeJacobianYOffset(true) == 32);
static_assert(TensorVolumeJacobianYSlot(false, 3, 3) + 1 ==
              tensor_volume_conventional_stride);
static_assert(TensorVolumeJacobianYSlot(true, 3, 4) + 1 ==
              tensor_volume_igr_common_stride);
static_assert(TensorVolumeIgrDqSlot(2, 2) + 1 ==
              tensor_volume_igr_stride);

} // namespace dg::hdg::runtime::cuda_w8a::detail

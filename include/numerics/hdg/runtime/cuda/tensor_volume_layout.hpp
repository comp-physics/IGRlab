#pragma once

/**
 * @file numerics/hdg/runtime/cuda/tensor_volume_layout.hpp
 * @brief Checked compact scratch layout for FP64 tensor volume assembly.
 *
 * This is immutable numerical-layout policy only.  It contains no run events,
 * iteration history, replay data, or observability interface.
 */

#include <cstddef>
#include <cstdint>
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

/** Exact bounded FP64 m8n8k4 work for one tensor-volume assembly call. */
struct TensorVolumeMmaWorkPlan final
{
  TensorVolumeScratchPlan scratch{};
  std::uint64_t contraction_launches = 0;
  std::uint64_t mma_tiles = 0;
  bool valid = false;
};

constexpr bool TensorVolumeCheckedAdd(
    const std::uint64_t left, const std::uint64_t right,
    std::uint64_t& result) noexcept
{
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    return false;
  result = left + right;
  return true;
}

constexpr bool TensorVolumeCheckedMultiply(
    const std::uint64_t left, const std::uint64_t right,
    std::uint64_t& result) noexcept
{
  if (left != 0 &&
      right > std::numeric_limits<std::uint64_t>::max() / left)
    return false;
  result = left * right;
  return true;
}

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

/**
 * Count every dynamic FP64 WMMA instruction executed by one successful
 * production tensor-volume assembly invocation.
 *
 * Each contraction launch covers one deterministic quadrature chunk.  One
 * block owns an (element,row-component) slab; its warps cover complete 8x8
 * output tiles and execute one m8n8k4 instruction for every phi/dx/dy feature
 * and every zero-padded group of four quadrature points.  Consequently a
 * short final chunk remains material work and is included exactly.
 */
constexpr TensorVolumeMmaWorkPlan PlanTensorVolumeMmaWork(
    const int elements, const int interior_components,
    const int scalar_interior_dofs, const int interior_dofs,
    const int trace_dofs, const int volume_points,
    const bool igr) noexcept
{
  TensorVolumeMmaWorkPlan plan{};
  plan.scratch = PlanTensorVolumeScratch(
      interior_dofs, trace_dofs, volume_points, igr);
  if (!plan.scratch.valid || elements <= 0 || interior_components <= 0 ||
      scalar_interior_dofs <= 0 || interior_dofs <= 0 ||
      trace_dofs <= 0 || volume_points <= 0)
    return plan;

  int remaining = volume_points;
  std::uint64_t quadrature_groups = 0;
  while (remaining > 0)
  {
    const int count = remaining < plan.scratch.chunk_points
        ? remaining : plan.scratch.chunk_points;
    if (count <= 0 || count > plan.scratch.capacity_points ||
        (remaining > count && count % tensor_volume_mma_inner != 0) ||
        plan.contraction_launches ==
            std::numeric_limits<std::uint64_t>::max())
      return TensorVolumeMmaWorkPlan{};
    const std::uint64_t groups =
        static_cast<std::uint64_t>((count - 1) /
                                   tensor_volume_mma_inner + 1);
    std::uint64_t updated_groups = 0;
    if (!TensorVolumeCheckedAdd(
            quadrature_groups, groups, updated_groups))
      return TensorVolumeMmaWorkPlan{};
    quadrature_groups = updated_groups;
    ++plan.contraction_launches;
    remaining -= count;
  }

  const std::uint64_t row_tiles =
      (static_cast<std::uint64_t>(scalar_interior_dofs) + 7U) / 8U;
  // The final output column is residual_r; all preceding columns are A.
  const std::uint64_t column_tiles =
      (static_cast<std::uint64_t>(interior_dofs) + 8U) / 8U;
  std::uint64_t work = static_cast<std::uint64_t>(elements);
  const std::uint64_t factors[] = {
      static_cast<std::uint64_t>(interior_components), row_tiles,
      column_tiles, static_cast<std::uint64_t>(tensor_volume_features),
      quadrature_groups};
  for (const std::uint64_t factor : factors)
  {
    std::uint64_t updated = 0;
    if (!TensorVolumeCheckedMultiply(work, factor, updated))
      return TensorVolumeMmaWorkPlan{};
    work = updated;
  }
  plan.mma_tiles = work;
  plan.valid = plan.contraction_launches > 0 && plan.mma_tiles > 0;
  return plan;
}

/**
 * Update the two constant-size tensor-volume aggregate values atomically with
 * respect to overflow checks.  Ordinary CUDA must retain exact zeros.
 */
constexpr bool AccumulateTensorVolumeWork(
    const bool tensor_selected, const std::uint64_t invocation_mma_tiles,
    std::uint64_t& volume_assembly_launches,
    std::uint64_t& volume_mma_tiles) noexcept
{
  if (!tensor_selected)
    return invocation_mma_tiles == 0 && volume_assembly_launches == 0 &&
        volume_mma_tiles == 0;
  if (invocation_mma_tiles == 0 ||
      volume_assembly_launches ==
          std::numeric_limits<std::uint64_t>::max() ||
      invocation_mma_tiles >
          std::numeric_limits<std::uint64_t>::max() - volume_mma_tiles)
    return false;
  ++volume_assembly_launches;
  volume_mma_tiles += invocation_mma_tiles;
  return true;
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

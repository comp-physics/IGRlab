#pragma once

/**
 * @file numerics/hdg/runtime/cuda/native_element_schwarz_plan.hpp
 * @brief CUDA-independent rank-one element-overlap Schwarz topology plan.
 *
 * The plan is built once while the persistent HDG topology is initialized.
 * It contains only immutable numerical indices: no solve events, iteration
 * history, replay data, or observability callbacks are retained.
 */

#include "numerics/hdg/generic/polynomial_order.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

inline constexpr int native_element_schwarz_faces = 4;
inline constexpr int native_element_schwarz_igr_trace_components = 5;
inline constexpr int native_element_schwarz_euler_trace_components = 4;
inline constexpr int native_element_schwarz_minimum_order =
    generic::minimum_polynomial_order;
inline constexpr int native_element_schwarz_maximum_order =
    generic::maximum_polynomial_order;
inline constexpr int native_element_schwarz_maximum_contributors = 2;
inline constexpr int native_element_schwarz_equilibration_passes = 2;

constexpr int NativeElementSchwarzPatchRows(
    const int order, const bool igr_active) noexcept
{
  return order >= native_element_schwarz_minimum_order &&
          order <= native_element_schwarz_maximum_order
      ? native_element_schwarz_faces *
            (igr_active ? native_element_schwarz_igr_trace_components
                        : native_element_schwarz_euler_trace_components) *
            (order + 1)
      : 0;
}

constexpr int NativeElementSchwarzIgrPatchRows(const int order) noexcept
{
  return NativeElementSchwarzPatchRows(order, true);
}

/**
 * Packed storage contract for the benchmark-only equilibrated field split.
 *
 * The exact unfactored principal patch remains in its existing n-by-n arena.
 * The equally large factor arena stores the Euler and sigma diagonal-block
 * factors, the scaled lower coupling, and two-sided FP64 scales.  This is
 * strictly smaller than n-by-n for every admitted IGR patch, so no
 * persistent allocation or alternate application lifecycle is required.
 */
struct NativeElementSchwarzFieldSplitLayout final
{
  int scalar_face_rows = 0;
  int patch_rows = 0;
  int euler_rows = 0;
  int sigma_rows = 0;
  std::size_t euler_lu_offset = 0;
  std::size_t sigma_lu_offset = 0;
  std::size_t lower_coupling_offset = 0;
  std::size_t left_scale_offset = 0;
  std::size_t right_scale_offset = 0;
  std::size_t required_doubles = 0;
  std::size_t warp_inverse_diagonal_offset = 0;
  std::size_t warp_rhs_permutation_offset = 0;
  std::size_t warp_required_doubles = 0;
  std::size_t available_doubles = 0;
  bool valid = false;
  bool warp_valid = false;
};

constexpr NativeElementSchwarzFieldSplitLayout
NativeElementSchwarzIgrFieldSplitLayout(const int order) noexcept
{
  NativeElementSchwarzFieldSplitLayout result{};
  const int patch_rows = NativeElementSchwarzIgrPatchRows(order);
  if (patch_rows == 0) return result;
  result.scalar_face_rows = order + 1;
  result.patch_rows = patch_rows;
  result.euler_rows = native_element_schwarz_faces *
      native_element_schwarz_euler_trace_components *
      result.scalar_face_rows;
  result.sigma_rows = native_element_schwarz_faces *
      result.scalar_face_rows;
  result.euler_lu_offset = 0;
  result.sigma_lu_offset = static_cast<std::size_t>(result.euler_rows) *
      result.euler_rows;
  result.lower_coupling_offset = result.sigma_lu_offset +
      static_cast<std::size_t>(result.sigma_rows) * result.sigma_rows;
  result.left_scale_offset = result.lower_coupling_offset +
      static_cast<std::size_t>(result.sigma_rows) * result.euler_rows;
  result.right_scale_offset = result.left_scale_offset + patch_rows;
  result.required_doubles = result.right_scale_offset + patch_rows;
  result.warp_inverse_diagonal_offset = result.required_doubles;
  result.warp_rhs_permutation_offset =
      result.warp_inverse_diagonal_offset + patch_rows;
  result.warp_required_doubles =
      result.warp_rhs_permutation_offset + patch_rows;
  result.available_doubles =
      static_cast<std::size_t>(patch_rows) * patch_rows;
  result.valid = result.euler_rows + result.sigma_rows == patch_rows &&
      result.required_doubles <= result.available_doubles;
  result.warp_valid = result.valid &&
      result.warp_required_doubles <= result.available_doubles;
  return result;
}

constexpr bool NativeElementSchwarzIgrPatchRowIsSigma(
    const int order, const int patch_row) noexcept
{
  const NativeElementSchwarzFieldSplitLayout layout =
      NativeElementSchwarzIgrFieldSplitLayout(order);
  if (!layout.valid || patch_row < 0 || patch_row >= layout.patch_rows)
    return false;
  const int face_span = native_element_schwarz_igr_trace_components *
      layout.scalar_face_rows;
  return (patch_row % face_span) /
      layout.scalar_face_rows ==
      native_element_schwarz_euler_trace_components;
}

/** Map a face-major/component-major patch row to [Euler, sigma] packing. */
constexpr int NativeElementSchwarzIgrFieldSplitPackedRow(
    const int order, const int patch_row) noexcept
{
  const NativeElementSchwarzFieldSplitLayout layout =
      NativeElementSchwarzIgrFieldSplitLayout(order);
  if (!layout.valid || patch_row < 0 || patch_row >= layout.patch_rows)
    return -1;
  const int face_span = native_element_schwarz_igr_trace_components *
      layout.scalar_face_rows;
  const int face = patch_row / face_span;
  const int within_face = patch_row - face * face_span;
  const int component = within_face / layout.scalar_face_rows;
  const int scalar = within_face - component * layout.scalar_face_rows;
  if (component < native_element_schwarz_euler_trace_components)
    return face * native_element_schwarz_euler_trace_components *
        layout.scalar_face_rows + component * layout.scalar_face_rows +
        scalar;
  return layout.euler_rows + face * layout.scalar_face_rows + scalar;
}

/** Inverse of NativeElementSchwarzIgrFieldSplitPackedRow. */
constexpr int NativeElementSchwarzIgrFieldSplitPatchRow(
    const int order, const int packed_row) noexcept
{
  const NativeElementSchwarzFieldSplitLayout layout =
      NativeElementSchwarzIgrFieldSplitLayout(order);
  if (!layout.valid || packed_row < 0 || packed_row >= layout.patch_rows)
    return -1;
  const int face_span = native_element_schwarz_igr_trace_components *
      layout.scalar_face_rows;
  if (packed_row < layout.euler_rows)
  {
    const int euler_face_span =
        native_element_schwarz_euler_trace_components *
        layout.scalar_face_rows;
    const int face = packed_row / euler_face_span;
    const int within_face = packed_row - face * euler_face_span;
    return face * face_span + within_face;
  }
  const int sigma = packed_row - layout.euler_rows;
  const int face = sigma / layout.scalar_face_rows;
  const int scalar = sigma - face * layout.scalar_face_rows;
  return face * face_span +
      native_element_schwarz_euler_trace_components *
          layout.scalar_face_rows + scalar;
}

constexpr int NativeElementSchwarzDecodedDof(const int encoded) noexcept
{
  return encoded >= 0 ? encoded : -1 - encoded;
}

constexpr int NativeElementSchwarzDofSign(const int encoded) noexcept
{
  return encoded >= 0 ? 1 : -1;
}

constexpr int NativeElementSchwarzEncodeDof(const int dof,
                                             const int sign) noexcept
{
  return dof < 0 || (sign != 1 && sign != -1)
      ? std::numeric_limits<int>::min()
      : (sign > 0 ? dof : -1 - dof);
}

struct NativeElementSchwarzPlanInput final
{
  int elements = 0;
  int patch_rows = 0;
  int local_rows = 0;
  int true_rows = 0;
  int matrix_nonzeros = 0;
  const int* signed_element_local_dofs = nullptr;
  std::size_t signed_element_local_dof_count = 0;
  const int* local_true_dofs = nullptr;
  std::size_t local_true_dof_count = 0;
  const int* matrix_sources = nullptr;
  const int* matrix_sections = nullptr;
  const int* matrix_targets = nullptr;
  std::size_t matrix_entry_count = 0;
};

/**
 * Rank-one matrix-free topology input.
 *
 * Unlike NativeElementSchwarzPlanInput, this contract does not depend on an
 * assembled true-CSR graph or on source/section/target staging arrays.  The
 * signed element-local map and the local-to-true map contain all topology
 * needed to recover the one or two element contributions to a principal
 * Schwarz-patch entry.
 */
struct NativeElementSchwarzDirectPlanInput final
{
  int elements = 0;
  int patch_rows = 0;
  int local_rows = 0;
  int true_rows = 0;
  const int* signed_element_local_dofs = nullptr;
  std::size_t signed_element_local_dof_count = 0;
  const int* local_true_dofs = nullptr;
  std::size_t local_true_dof_count = 0;
};

struct NativeElementSchwarzPlan final
{
  std::vector<int> signed_element_true_dofs{};
  std::vector<int> first_element_sources{};
  std::vector<int> second_element_sources{};
  std::vector<int> true_multiplicity{};
  // Exactly two fixed slots per true row. -1 denotes an absent boundary
  // incidence. This supports deterministic scatter without device atomics.
  std::vector<int> true_incidence_slots{};
  int patch_rows = 0;
  int minimum_multiplicity = 0;
  int maximum_multiplicity = 0;
  int maximum_contributors = 0;
  bool valid = false;
};

/**
 * Build the rank-one element-overlap plan directly from trace incidence.
 *
 * Every admitted true trace row has one boundary incidence or two interior
 * incidences.  The contributors to an ordered true entry (i,j) are exactly
 * the elements common to the two incidence lists.  Incidences are recorded
 * in element-major/local-row-major order, so the recovered source order is
 * identical to the legacy CSR-derived plan's nested element/row/column order.
 */
inline NativeElementSchwarzPlan BuildDirectNativeElementSchwarzPlan(
    const NativeElementSchwarzDirectPlanInput& input)
{
  NativeElementSchwarzPlan result{};
  if (input.elements <= 0 || input.patch_rows <= 0 ||
      input.local_rows <= 0 || input.true_rows <= 0 ||
      input.signed_element_local_dofs == nullptr ||
      input.local_true_dofs == nullptr)
    return result;

  const std::size_t elements = static_cast<std::size_t>(input.elements);
  const std::size_t rows = static_cast<std::size_t>(input.patch_rows);
  if (rows > std::numeric_limits<std::size_t>::max() / elements)
    return result;
  const std::size_t element_rows = elements * rows;
  if (rows > std::numeric_limits<std::size_t>::max() / element_rows)
    return result;
  const std::size_t element_entries = element_rows * rows;
  if (element_rows > static_cast<std::size_t>(
                         std::numeric_limits<int>::max()) ||
      element_entries > static_cast<std::size_t>(
                            std::numeric_limits<int>::max()) ||
      input.signed_element_local_dof_count != element_rows ||
      input.local_true_dof_count < static_cast<std::size_t>(input.local_rows))
    return result;

  try
  {
    result.signed_element_true_dofs.assign(element_rows, 0);
    result.first_element_sources.assign(element_entries, -1);
    result.second_element_sources.assign(element_entries, -1);
    result.true_multiplicity.assign(
        static_cast<std::size_t>(input.true_rows), 0);
    result.true_incidence_slots.assign(
        2 * static_cast<std::size_t>(input.true_rows), -1);

    for (std::size_t slot = 0; slot < element_rows; ++slot)
    {
      const int encoded_local = input.signed_element_local_dofs[slot];
      const int local = NativeElementSchwarzDecodedDof(encoded_local);
      if (local < 0 || local >= input.local_rows) return result;
      const int truth = input.local_true_dofs[static_cast<std::size_t>(local)];
      if (truth < 0 || truth >= input.true_rows) return result;
      const int encoded_true = NativeElementSchwarzEncodeDof(
          truth, NativeElementSchwarzDofSign(encoded_local));
      if (encoded_true == std::numeric_limits<int>::min()) return result;
      result.signed_element_true_dofs[slot] = encoded_true;

      int& multiplicity =
          result.true_multiplicity[static_cast<std::size_t>(truth)];
      if (multiplicity < 0 ||
          multiplicity >= native_element_schwarz_maximum_contributors)
        return result;
      if (multiplicity > 0)
      {
        const int previous = result.true_incidence_slots[
            2 * static_cast<std::size_t>(truth)];
        if (previous < 0 ||
            previous / input.patch_rows ==
                static_cast<int>(slot) / input.patch_rows)
          return result;
      }
      result.true_incidence_slots[
          2 * static_cast<std::size_t>(truth) +
          static_cast<std::size_t>(multiplicity)] = static_cast<int>(slot);
      ++multiplicity;
    }

    const auto multiplicity_bounds = std::minmax_element(
        result.true_multiplicity.begin(), result.true_multiplicity.end());
    if (multiplicity_bounds.first == result.true_multiplicity.end() ||
        *multiplicity_bounds.first <= 0 ||
        *multiplicity_bounds.second >
            native_element_schwarz_maximum_contributors)
      return result;
    result.minimum_multiplicity = *multiplicity_bounds.first;
    result.maximum_multiplicity = *multiplicity_bounds.second;

    for (std::size_t target = 0; target < element_entries; ++target)
    {
      const std::size_t target_element = target / (rows * rows);
      const std::size_t target_entry = target - target_element * rows * rows;
      const std::size_t target_row = target_element * rows +
          target_entry / rows;
      const std::size_t target_column = target_element * rows +
          target_entry % rows;
      const int row_truth = NativeElementSchwarzDecodedDof(
          result.signed_element_true_dofs[target_row]);
      const int column_truth = NativeElementSchwarzDecodedDof(
          result.signed_element_true_dofs[target_column]);

      int first = -1;
      int second = -1;
      const int row_multiplicity = result.true_multiplicity[
          static_cast<std::size_t>(row_truth)];
      const int column_multiplicity = result.true_multiplicity[
          static_cast<std::size_t>(column_truth)];
      for (int row_incidence = 0; row_incidence < row_multiplicity;
           ++row_incidence)
      {
        const int source_row = result.true_incidence_slots[
            2 * static_cast<std::size_t>(row_truth) +
            static_cast<std::size_t>(row_incidence)];
        if (source_row < 0) return result;
        const int source_element = source_row / input.patch_rows;
        for (int column_incidence = 0;
             column_incidence < column_multiplicity; ++column_incidence)
        {
          const int source_column = result.true_incidence_slots[
              2 * static_cast<std::size_t>(column_truth) +
              static_cast<std::size_t>(column_incidence)];
          if (source_column < 0) return result;
          if (source_column / input.patch_rows != source_element) continue;
          const int source = source_row * input.patch_rows +
              source_column % input.patch_rows;
          if (first < 0)
            first = source;
          else if (second < 0 && first != source)
            second = source;
          else
            return result;
        }
      }
      if (first < 0) return result;
      if (second >= 0 && second < first) std::swap(first, second);
      result.first_element_sources[target] = first;
      result.second_element_sources[target] = second;
      result.maximum_contributors = std::max(
          result.maximum_contributors, second >= 0 ? 2 : 1);
    }

    result.valid = result.maximum_contributors >= 1 &&
        result.maximum_contributors <=
            native_element_schwarz_maximum_contributors;
    if (result.valid) result.patch_rows = input.patch_rows;
  }
  catch (...)
  {
    return NativeElementSchwarzPlan{};
  }
  return result;
}

/**
 * Build the exact source plan for every entry of every principal patch.
 *
 * The persistent rank-one matrix staging plan already maps each signed
 * element-Schur entry to its assembled true-CSR entry.  Reversing that map
 * yields the one or two retained element contributions whose signed sum is
 * one global matrix entry.  A target element patch applies its own row/column
 * orientation when those contributions are materialized on the device,
 * producing exactly R_e A R_e^T without reading HYPRE values.
 */
inline NativeElementSchwarzPlan BuildNativeElementSchwarzPlan(
    const NativeElementSchwarzPlanInput& input)
{
  NativeElementSchwarzPlan result{};
  if (input.elements <= 0 || input.patch_rows <= 0 ||
      input.local_rows <= 0 || input.true_rows <= 0 ||
      input.matrix_nonzeros <= 0 ||
      input.signed_element_local_dofs == nullptr ||
      input.local_true_dofs == nullptr || input.matrix_sources == nullptr ||
      input.matrix_sections == nullptr || input.matrix_targets == nullptr)
    return result;

  const std::size_t elements = static_cast<std::size_t>(input.elements);
  const std::size_t rows = static_cast<std::size_t>(input.patch_rows);
  if (rows > std::numeric_limits<std::size_t>::max() / elements)
    return result;
  const std::size_t element_rows = elements * rows;
  if (rows > std::numeric_limits<std::size_t>::max() / element_rows)
    return result;
  const std::size_t element_entries = element_rows * rows;
  if (input.signed_element_local_dof_count != element_rows ||
      input.local_true_dof_count < static_cast<std::size_t>(input.local_rows) ||
      input.matrix_entry_count != element_entries)
    return result;

  try
  {
    result.signed_element_true_dofs.assign(element_rows, 0);
    result.first_element_sources.assign(element_entries, -1);
    result.second_element_sources.assign(element_entries, -1);
    result.true_multiplicity.assign(
        static_cast<std::size_t>(input.true_rows), 0);
    result.true_incidence_slots.assign(
        2 * static_cast<std::size_t>(input.true_rows), -1);
    std::vector<int> source_target(element_entries, -1);
    std::vector<int> first_by_target(
        static_cast<std::size_t>(input.matrix_nonzeros), -1);
    std::vector<int> second_by_target(
        static_cast<std::size_t>(input.matrix_nonzeros), -1);

    for (std::size_t slot = 0; slot < element_rows; ++slot)
    {
      const int encoded_local = input.signed_element_local_dofs[slot];
      const int local = NativeElementSchwarzDecodedDof(encoded_local);
      if (local < 0 || local >= input.local_rows) return result;
      const int truth = input.local_true_dofs[static_cast<std::size_t>(local)];
      if (truth < 0 || truth >= input.true_rows) return result;
      const int encoded_true = NativeElementSchwarzEncodeDof(
          truth, NativeElementSchwarzDofSign(encoded_local));
      if (encoded_true == std::numeric_limits<int>::min()) return result;
      result.signed_element_true_dofs[slot] = encoded_true;
      int& multiplicity =
          result.true_multiplicity[static_cast<std::size_t>(truth)];
      if (multiplicity < 0 ||
          multiplicity >= native_element_schwarz_maximum_contributors)
        return result;
      result.true_incidence_slots[
          2 * static_cast<std::size_t>(truth) +
          static_cast<std::size_t>(multiplicity)] =
              static_cast<int>(slot);
      ++multiplicity;
    }

    for (std::size_t entry = 0; entry < element_entries; ++entry)
    {
      const int source = input.matrix_sources[entry];
      const int section = input.matrix_sections[entry];
      const int target = input.matrix_targets[entry];
      if (source < 0 || static_cast<std::size_t>(source) >= element_entries ||
          section != 0 || target < 0 || target >= input.matrix_nonzeros ||
          source_target[static_cast<std::size_t>(source)] != -1)
        return result;
      source_target[static_cast<std::size_t>(source)] = target;
      int& first = first_by_target[static_cast<std::size_t>(target)];
      int& second = second_by_target[static_cast<std::size_t>(target)];
      if (first < 0)
        first = source;
      else if (second < 0 && first != source)
        second = source;
      else
        return result;
    }

    for (std::size_t target_source = 0;
         target_source < element_entries; ++target_source)
    {
      const int target = source_target[target_source];
      if (target < 0) return result;
      const int first = first_by_target[static_cast<std::size_t>(target)];
      const int second = second_by_target[static_cast<std::size_t>(target)];
      if (first < 0) return result;
      result.first_element_sources[target_source] = first;
      result.second_element_sources[target_source] = second;
      result.maximum_contributors = std::max(
          result.maximum_contributors, second >= 0 ? 2 : 1);
    }

    const auto multiplicity_bounds = std::minmax_element(
        result.true_multiplicity.begin(), result.true_multiplicity.end());
    if (multiplicity_bounds.first == result.true_multiplicity.end() ||
        *multiplicity_bounds.first <= 0 ||
        *multiplicity_bounds.second >
            native_element_schwarz_maximum_contributors)
      return result;
    result.minimum_multiplicity = *multiplicity_bounds.first;
    result.maximum_multiplicity = *multiplicity_bounds.second;
    result.valid = result.maximum_contributors >= 1 &&
        result.maximum_contributors <=
            native_element_schwarz_maximum_contributors;
    if (result.valid) result.patch_rows = input.patch_rows;
  }
  catch (...)
  {
    return NativeElementSchwarzPlan{};
  }
  return result;
}

} // namespace dg::hdg::runtime::cuda_w8a::detail

#pragma once

/**
 * @file numerics/hdg/runtime/cuda/native_element_schwarz_plan.hpp
 * @brief CUDA-independent rank-one element-overlap Schwarz topology plan.
 *
 * The plan is built once while the persistent HDG topology is initialized.
 * It contains only immutable numerical indices: no solve events, iteration
 * history, replay data, or observability callbacks are retained.
 */

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

inline constexpr int native_element_schwarz_faces = 4;
inline constexpr int native_element_schwarz_igr_trace_components = 5;
inline constexpr int native_element_schwarz_minimum_order = 1;
inline constexpr int native_element_schwarz_maximum_order = 3;
inline constexpr int native_element_schwarz_maximum_contributors = 2;

constexpr int NativeElementSchwarzIgrPatchRows(const int order) noexcept
{
  return order >= native_element_schwarz_minimum_order &&
          order <= native_element_schwarz_maximum_order
      ? native_element_schwarz_faces *
            native_element_schwarz_igr_trace_components * (order + 1)
      : 0;
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

struct NativeElementSchwarzPlan final
{
  std::vector<int> signed_element_true_dofs{};
  std::vector<int> first_element_sources{};
  std::vector<int> second_element_sources{};
  std::vector<int> true_multiplicity{};
  // Exactly two fixed slots per true row. -1 denotes an absent boundary
  // incidence. This supports deterministic scatter without device atomics.
  std::vector<int> true_incidence_slots{};
  int minimum_multiplicity = 0;
  int maximum_multiplicity = 0;
  int maximum_contributors = 0;
  bool valid = false;
};

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
  }
  catch (...)
  {
    return NativeElementSchwarzPlan{};
  }
  return result;
}

} // namespace dg::hdg::runtime::cuda_w8a::detail

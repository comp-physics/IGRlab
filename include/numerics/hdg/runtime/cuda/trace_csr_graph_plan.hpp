#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

struct TraceCsrGraphPlanInput final
{
  int elements = 0;
  int element_rows = 0;
  int local_rows = 0;
  const int* signed_element_local_dofs = nullptr;
  std::size_t signed_element_local_dof_count = 0;
};

/**
 * Persistent local trace graph in diagonal-first CSR order.
 *
 * The graph is the signed union of the dense trace blocks induced by every
 * element. Numerical values are absent because setup overwrites every graph
 * value before assembly into the true system.
 */
struct TraceCsrGraphPlan final
{
  std::vector<int> offsets{};
  std::vector<int> columns{};
  bool legacy_columns_sorted = false;
  bool valid = false;
};

inline int TraceCsrDecodedDof(const int encoded) noexcept
{
  if (encoded == std::numeric_limits<int>::min()) return -1;
  return encoded >= 0 ? encoded : -1 - encoded;
}

/**
 * Construct the exact element-dense graph without MFEM's incremental
 * linked-list insertion. Signed duplicate contributions and cancellation are
 * retained. MFEM's linked list emits columns in reverse first-insertion
 * order; each row reproduces that order and then moves its diagonal first.
 */
inline TraceCsrGraphPlan BuildTraceCsrGraphPlan(
    const TraceCsrGraphPlanInput& input)
{
  TraceCsrGraphPlan result{};
  if (input.elements <= 0 || input.element_rows <= 0 ||
      input.local_rows <= 0 || input.signed_element_local_dofs == nullptr)
    return result;

  const std::size_t elements = static_cast<std::size_t>(input.elements);
  const std::size_t element_rows =
      static_cast<std::size_t>(input.element_rows);
  if (element_rows > std::numeric_limits<std::size_t>::max() / elements)
    return result;
  const std::size_t element_row_count = elements * element_rows;
  if (input.signed_element_local_dof_count != element_row_count ||
      element_row_count >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      element_rows > std::numeric_limits<std::size_t>::max() /
          element_row_count)
    return result;
  const std::size_t dense_entry_upper_bound =
      element_row_count * element_rows;

  try
  {
    std::vector<int> decoded_element_dofs(element_row_count, -1);
    std::vector<int> incidence_offsets(
        static_cast<std::size_t>(input.local_rows) + 1, 0);
    for (std::size_t slot = 0; slot < element_row_count; ++slot)
    {
      const int local =
          TraceCsrDecodedDof(input.signed_element_local_dofs[slot]);
      if (local < 0 || local >= input.local_rows) return result;
      decoded_element_dofs[slot] = local;
      int& count = incidence_offsets[static_cast<std::size_t>(local) + 1];
      if (count == std::numeric_limits<int>::max()) return result;
      ++count;
    }
    for (int row = 0; row < input.local_rows; ++row)
    {
      const int count = incidence_offsets[static_cast<std::size_t>(row) + 1];
      const int begin = incidence_offsets[static_cast<std::size_t>(row)];
      if (count > std::numeric_limits<int>::max() - begin) return result;
      incidence_offsets[static_cast<std::size_t>(row) + 1] = begin + count;
    }
    if (incidence_offsets.back() != static_cast<int>(element_row_count))
      return result;

    std::vector<int> incidence_slots(element_row_count, -1);
    std::vector<int> incidence_cursor = incidence_offsets;
    for (std::size_t slot = 0; slot < element_row_count; ++slot)
    {
      const int local = decoded_element_dofs[slot];
      int& cursor = incidence_cursor[static_cast<std::size_t>(local)];
      incidence_slots[static_cast<std::size_t>(cursor++)] =
          static_cast<int>(slot);
    }

    result.offsets.assign(
        static_cast<std::size_t>(input.local_rows) + 1, 0);
    result.columns.reserve(std::min(
        dense_entry_upper_bound,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    std::vector<int> column_position(
        static_cast<std::size_t>(input.local_rows), -1);
    std::vector<int> first_seen_columns;
    std::vector<int> signed_sums;
    std::vector<int> legacy_columns;
    bool legacy_columns_sorted = true;
    for (int row = 0; row < input.local_rows; ++row)
    {
      const int incidence_begin =
          incidence_offsets[static_cast<std::size_t>(row)];
      const int incidence_end =
          incidence_offsets[static_cast<std::size_t>(row) + 1];
      if (incidence_begin >= incidence_end) return result;
      const std::size_t incidence_count =
          static_cast<std::size_t>(incidence_end - incidence_begin);
      if (element_rows > std::numeric_limits<std::size_t>::max() /
          incidence_count)
        return result;

      first_seen_columns.clear();
      signed_sums.clear();
      first_seen_columns.reserve(incidence_count * element_rows);
      signed_sums.reserve(incidence_count * element_rows);
      for (int incidence = incidence_begin; incidence < incidence_end;
           ++incidence)
      {
        const int row_slot = incidence_slots[
            static_cast<std::size_t>(incidence)];
        if (row_slot < 0 ||
            static_cast<std::size_t>(row_slot) >= element_row_count ||
            decoded_element_dofs[static_cast<std::size_t>(row_slot)] != row)
          return result;
        const int row_sign = input.signed_element_local_dofs[
            static_cast<std::size_t>(row_slot)] >= 0 ? 1 : -1;
        const std::size_t element =
            static_cast<std::size_t>(row_slot) / element_rows;
        const std::size_t base = element * element_rows;
        for (std::size_t slot = base; slot < base + element_rows; ++slot)
        {
          const int column = decoded_element_dofs[slot];
          const int column_sign =
              input.signed_element_local_dofs[slot] >= 0 ? 1 : -1;
          const int contribution = row_sign * column_sign;
          int& position =
              column_position[static_cast<std::size_t>(column)];
          if (position < 0)
          {
            if (first_seen_columns.size() >= static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
              return result;
            position = static_cast<int>(first_seen_columns.size());
            first_seen_columns.push_back(column);
            signed_sums.push_back(contribution);
          }
          else
          {
            int& sum = signed_sums[static_cast<std::size_t>(position)];
            if ((contribution > 0 &&
                 sum == std::numeric_limits<int>::max()) ||
                (contribution < 0 &&
                 sum == std::numeric_limits<int>::min()))
              return result;
            sum += contribution;
          }
        }
      }

      legacy_columns.clear();
      legacy_columns.reserve(first_seen_columns.size());
      for (std::size_t reverse = first_seen_columns.size(); reverse > 0;
           --reverse)
        if (signed_sums[reverse - 1] != 0)
          legacy_columns.push_back(first_seen_columns[reverse - 1]);
      const auto diagonal = std::find(
          legacy_columns.begin(), legacy_columns.end(), row);
      if (diagonal == legacy_columns.end() ||
          result.columns.size() > static_cast<std::size_t>(
              std::numeric_limits<int>::max()) ||
          legacy_columns.size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) -
                  result.columns.size())
        return result;

      int previous = -1;
      for (const int column : legacy_columns)
      {
        legacy_columns_sorted = legacy_columns_sorted && previous <= column;
        previous = column;
      }
      result.offsets[static_cast<std::size_t>(row)] =
          static_cast<int>(result.columns.size());
      result.columns.push_back(row);
      result.columns.insert(result.columns.end(), legacy_columns.begin(),
                            diagonal);
      result.columns.insert(result.columns.end(), diagonal + 1,
                            legacy_columns.end());
      for (const int column : first_seen_columns)
        column_position[static_cast<std::size_t>(column)] = -1;
    }
    result.offsets.back() = static_cast<int>(result.columns.size());
    result.legacy_columns_sorted = legacy_columns_sorted;
    result.valid = result.offsets.front() == 0 &&
        result.offsets.back() > 0;
  }
  catch (...)
  {
    return TraceCsrGraphPlan{};
  }
  return result;
}

}  // namespace dg::hdg::runtime::cuda_w8a::detail

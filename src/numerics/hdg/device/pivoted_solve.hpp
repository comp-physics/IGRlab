#pragma once

#include <mfem.hpp>

namespace dg::hdg::device::detail
{
MFEM_HOST_DEVICE inline void SolvePivoted(const double* lu,
                                          const double* pivots,
                                          const int rows, double* values,
                                          const int columns) noexcept
{
  for (int row = 0; row < rows; ++row)
  {
    const int pivot = static_cast<int>(pivots[row]);
    if (pivot != row)
      for (int column = 0; column < columns; ++column)
      {
        const double temporary = values[row * columns + column];
        values[row * columns + column] = values[pivot * columns + column];
        values[pivot * columns + column] = temporary;
      }
  }
  for (int row = 0; row < rows; ++row)
    for (int below = row + 1; below < rows; ++below)
      for (int column = 0; column < columns; ++column)
      {
        values[below * columns + column] -=
            lu[below * rows + row] * values[row * columns + column];
      }
  for (int row = rows; row-- > 0;)
    for (int column = 0; column < columns; ++column)
    {
      for (int right = row + 1; right < rows; ++right)
      {
        values[row * columns + column] -=
            lu[row * rows + right] * values[right * columns + column];
      }
      values[row * columns + column] /= lu[row * rows + row];
    }
}
}  // namespace dg::hdg::device::detail

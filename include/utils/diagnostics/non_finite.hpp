#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "mfem.hpp"

/**
 * @file utils/diagnostics/non_finite.hpp
 * @brief Helpers to detect NaN/inf values in solution vectors.
 * @ingroup utils_config
 */

namespace mfem
{
namespace dg_igr
{

/**
 * @brief Summary of non-finite values in a distributed vector.
 */
struct NonFiniteSummary
{
  int global_non_finite = 0; ///< Total NaN/inf entries across all ranks.
  double global_min = std::numeric_limits<
    double>::infinity(); ///< Global min over finite entries.
  double global_max = -std::numeric_limits<
    double>::infinity(); ///< Global max over finite entries.
};

/**
 * @brief Scan a vector for NaN/inf values and reduce the result across ranks.
 *
 * @param[in] v    Vector to scan (local portion on this rank).
 * @param[in] comm MPI communicator used for global reductions.
 * @return Summary with global non-finite count and finite min/max.
 */
inline NonFiniteSummary ScanNonFinite(const mfem::Vector& v, MPI_Comm comm)
{
  double local_min = std::numeric_limits<double>::infinity();
  double local_max = -std::numeric_limits<double>::infinity();
  int local_non_finite = 0;

  for (int i = 0; i < v.Size(); ++i)
  {
    const double value = v(i);
    if (std::isfinite(value))
    {
      local_min = std::min(local_min, value);
      local_max = std::max(local_max, value);
    }
    else
    {
      ++local_non_finite;
    }
  }

  NonFiniteSummary summary;
  MPI_Allreduce(&local_non_finite,
                &summary.global_non_finite,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_SUM,
                comm);
  MPI_Allreduce(&local_min,
                &summary.global_min,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MIN,
                comm);
  MPI_Allreduce(&local_max,
                &summary.global_max,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MAX,
                comm);
  return summary;
}

/**
 * @brief Scan a ParGridFunction for NaN/inf values using its communicator.
 * @param[in] gf Grid function to scan.
 * @return Summary with global non-finite count and finite min/max.
 */
inline NonFiniteSummary ScanNonFinite(const mfem::ParGridFunction& gf)
{
  const mfem::ParFiniteElementSpace* pfes = gf.ParFESpace();
  MFEM_VERIFY(pfes != nullptr,
              "ParGridFunction must have a ParFiniteElementSpace.");
  return ScanNonFinite(static_cast<const mfem::Vector&>(gf), pfes->GetComm());
}

} // namespace dg_igr
} // namespace mfem

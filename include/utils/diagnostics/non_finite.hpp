#pragma once

#include "igrlab/compat.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "mfem.hpp"

/**
 * @file utils/diagnostics/non_finite.hpp
 * @brief Helpers to detect NaN/inf values in solution vectors.
 * @ingroup utils_config
 */

namespace mfem
{
namespace igrlab
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
 * @brief First non-finite entry found in a local vector.
 */
struct FirstNonFiniteEntry
{
  bool found = false;
  int local_index = -1;
  int scalar_dof = -1;
  int component = -1;
  double value = 0.0;
};

/**
 * @brief Detailed non-finite scan summary for a distributed vector.
 */
struct DetailedNonFiniteSummary
{
  NonFiniteSummary summary;
  int first_bad_rank = -1;
  std::vector<int> global_component_non_finite;
  FirstNonFiniteEntry first_local_entry;
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
 * @brief Find the first local NaN/inf entry and map it to scalar dof/component.
 */
inline FirstNonFiniteEntry
FindFirstNonFiniteEntry(const mfem::Vector& v,
                        int vdim,
                        mfem::Ordering::Type ordering)
{
  FirstNonFiniteEntry entry;
  const int scalar_ndofs =
    (vdim > 0 && v.Size() % vdim == 0) ? (v.Size() / vdim) : -1;

  for (int i = 0; i < v.Size(); ++i)
  {
    const double value = v(i);
    if (!std::isfinite(value))
    {
      entry.found = true;
      entry.local_index = i;
      entry.value = value;
      if (scalar_ndofs > 0)
      {
        if (ordering == mfem::Ordering::byNODES)
        {
          entry.scalar_dof = i / vdim;
          entry.component = i % vdim;
        }
        else
        {
          entry.scalar_dof = i % scalar_ndofs;
          entry.component = i / scalar_ndofs;
        }
      }
      return entry;
    }
  }

  return entry;
}

/**
 * @brief Find the lowest MPI rank that observed a non-finite entry.
 */
inline int FindFirstBadRank(bool found, MPI_Comm comm)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int first_bad_rank = found ? rank : std::numeric_limits<int>::max();
  MPI_Allreduce(MPI_IN_PLACE,
                &first_bad_rank,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_MIN,
                comm);
  if (first_bad_rank == std::numeric_limits<int>::max())
  {
    return -1;
  }
  return first_bad_rank;
}

/**
 * @brief Scan a distributed vector and keep component-level counts and first bad
 *        local entry metadata.
 */
inline DetailedNonFiniteSummary
ScanNonFiniteDetailed(const mfem::Vector& v,
                      MPI_Comm comm,
                      int vdim,
                      mfem::Ordering::Type ordering)
{
  DetailedNonFiniteSummary out;
  out.summary = ScanNonFinite(v, comm);
  out.first_local_entry = FindFirstNonFiniteEntry(v, vdim, ordering);
  out.first_bad_rank = FindFirstBadRank(out.first_local_entry.found, comm);

  if (vdim <= 0)
  {
    return out;
  }

  const int scalar_ndofs =
    (v.Size() % vdim == 0) ? (v.Size() / vdim) : -1;
  if (scalar_ndofs <= 0)
  {
    return out;
  }

  std::vector<int> local_counts(vdim, 0);
  for (int i = 0; i < v.Size(); ++i)
  {
    const double value = v(i);
    if (std::isfinite(value))
    {
      continue;
    }

    const int component =
      (ordering == mfem::Ordering::byNODES) ? (i % vdim) : (i / scalar_ndofs);
    if (component >= 0 && component < vdim)
    {
      ++local_counts[component];
    }
  }

  out.global_component_non_finite.resize(vdim, 0);
  MPI_Allreduce(local_counts.data(),
                out.global_component_non_finite.data(),
                vdim,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_SUM,
                comm);
  return out;
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

} // namespace igrlab
} // namespace mfem

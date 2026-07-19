#pragma once

#include <mfem.hpp>

#include <string>
#include <vector>

namespace dg::hdg
{

enum class ScalarProblem
{
   poisson,
   advection
};

enum class MeshGeometry
{
   affine,
   curved
};

struct Comparison
{
   std::string name;
   double value = 0.0;
   double tolerance = 0.0;
   bool pass = false;
};

struct ConvergenceRow
{
   ScalarProblem problem = ScalarProblem::poisson;
   MeshGeometry geometry = MeshGeometry::affine;
   int order = 1;
   int elements = 0;
   double h_max = 0.0;
   double l2_error = 0.0;
   double rate = 0.0;
   double recovery_error = 0.0;
   double conservation_error = 0.0;
};

struct SerialEvidence
{
   std::vector<Comparison> comparisons;
   std::vector<ConvergenceRow> convergence;
   bool pass = false;
};

struct MpiEvidence
{
   int ranks = 1;
   int global_elements = 0;
   int mfem_partition_elements = 0;
   int owned_trace_dofs = 0;
   std::vector<Comparison> comparisons;
   bool pass = false;
};

/// Run deterministic algebra, orientation, conservation, boundary, recovery,
/// and p1-p4 manufactured-solution checks. No production miniapp state is used.
SerialEvidence RunSerialOracle();

/// Check deterministic skeleton ownership and exchange with one or two ranks.
/// The caller owns MPI initialization and the external nonblocking lease.
MpiEvidence RunMpiOracle(MPI_Comm comm);

const char *ToString(ScalarProblem problem);
const char *ToString(MeshGeometry geometry);

}  // namespace dg::hdg

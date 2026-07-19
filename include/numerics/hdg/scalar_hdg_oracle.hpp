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

struct ConvergenceGate
{
   ScalarProblem problem = ScalarProblem::poisson;
   MeshGeometry geometry = MeshGeometry::affine;
   int order = 1;
   double observed_final_rate = 0.0;
   double required_final_rate = 0.0;
   double margin = 0.0;
   bool all_values_finite = false;
   bool errors_strictly_decrease = false;
   bool pass = false;
};

struct SerialEvidence
{
   std::vector<Comparison> comparisons;
   std::vector<ConvergenceRow> convergence;
   std::vector<ConvergenceGate> convergence_gates;
   bool pass = false;
};

struct MpiEvidence
{
   int ranks = 1;
   int global_elements = 0;
   int mfem_partition_elements = 0;
   int owned_trace_dofs = 0;
   int ghost_trace_dofs = 0;
   int shared_faces = 0;
   std::vector<int> map_rank;
   std::vector<int> map_local_dof;
   std::vector<long long> map_global_true_dof;
   std::vector<int> map_owner_rank;
   std::vector<int> map_is_owned;
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

#pragma once

#include <mpi.h>

#include <vector>

namespace dg_igr_hdg_cpu_test
{

struct IndependentOracleResult
{
  std::vector<double> interior;
  std::vector<double> true_trace;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double maximum_true_residual = 0.0;
  int newton_updates = 0;
  bool mesh_rebuilt = false;
  bool analytic_source_flux_jacobian = false;
  bool signed_maps_rebuilt = false;
  bool local_schur_rebuilt = false;
  bool global_ptsp_rebuilt = false;
  bool true_residual_rebuilt = false;
  bool distribute_rebuilt = false;
  bool recovery_rebuilt = false;
  bool admissible = false;
  bool passed = false;
};

/** Independent test-local p1/N3 first-step oracle; no production helpers. */
bool RunIndependentFirstStep(MPI_Comm communicator,
                             IndependentOracleResult& result);

} // namespace dg_igr_hdg_cpu_test

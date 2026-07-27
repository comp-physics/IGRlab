#pragma once

#include <mpi.h>

#include <array>
#include <cstdint>
#include <vector>

namespace dg_igr_hdg_cpu_test
{

struct IndependentLinearizationHashes
{
  std::array<std::uint8_t, 32> residual{};
  std::array<std::uint8_t, 32> matrix{};
  std::array<std::uint8_t, 32> factor{};
  std::array<std::uint8_t, 32> solution{};
  std::array<std::uint8_t, 32> candidate{};
};

struct IndependentOracleResult
{
  std::vector<double> interior;
  std::vector<double> true_trace;
  std::vector<IndependentLinearizationHashes> linearizations;
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

#include "sc25_scalar_igr_independent_oracle.hpp"
#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  mfem::Hypre::Init();
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  namespace sc25 = dg::hdg::runtime::cpu::sc25;
  int failures = 0;
  if (ranks != 1)
  {
    if (rank == 0) { std::cerr << "SC25 harness requires exactly one rank\n"; }
    failures = 1;
  }
  else
  {
    sc25::FoundationConfig config;
    config.polynomial_degree = 1;
    config.elements_per_direction = 8;
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> foundation;
    const sc25::FoundationStatus status =
        sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
            MPI_COMM_WORLD, sc25::CpuSerialTag{}, config, foundation);
    if (status != sc25::FoundationStatus::ready || !foundation)
    {
      std::cerr << "rank 0: MFEM-native SC25 C0 creation failed\n";
      failures = 1;
    }
    else
    {
      dg_igr_hdg_sc25_cpu_test::IndependentC0Evidence evidence;
      if (!dg_igr_hdg_sc25_cpu_test::RunIndependentC0Oracle(
              *foundation, evidence))
      {
        failures = static_cast<int>(evidence.failures.size());
        for (const std::string& failure : evidence.failures)
        {
          std::cerr << "rank 0: " << failure << '\n';
        }
      }
      else
      {
        std::cout << "PASS: MFEM-native SC25 C0 foundation ("
                  << evidence.checked_elements << " elements, "
                  << evidence.checked_restriction_entries
                  << " restriction entries)\n";
      }
    }
    sc25::FoundationConfig rejected_periodic_config;
    rejected_periodic_config.polynomial_degree = 1;
    rejected_periodic_config.elements_per_direction = 2;
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>
        rejected_periodic_foundation;
    const sc25::FoundationStatus rejected_periodic_status =
        sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
            MPI_COMM_WORLD, sc25::CpuSerialTag{}, rejected_periodic_config,
            rejected_periodic_foundation);
    if (rejected_periodic_status !=
            sc25::FoundationStatus::invalid_argument ||
        rejected_periodic_foundation)
    {
      std::cerr << "rank 0: N<3 periodic mesh was not rejected safely\n";
      ++failures;
    }

    sc25::FoundationConfig coupled_config;
    coupled_config.polynomial_degree = 1;
    coupled_config.elements_per_direction = 3;
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>
        coupled_foundation;
    const sc25::FoundationStatus coupled_status =
        sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
            MPI_COMM_WORLD, sc25::CpuSerialTag{}, coupled_config,
            coupled_foundation);
    if (coupled_status != sc25::FoundationStatus::ready ||
        !coupled_foundation)
    {
      std::cerr << "rank 0: MFEM-native SC25 C1 creation failed\n";
      ++failures;
    }
    else
    {
      dg_igr_hdg_sc25_cpu_test::IndependentC1Evidence coupled_evidence;
      if (!dg_igr_hdg_sc25_cpu_test::RunIndependentC1Oracle(
              *coupled_foundation, coupled_evidence))
      {
        failures += std::max(
            1, static_cast<int>(coupled_evidence.failures.size()));
        for (const std::string& failure : coupled_evidence.failures)
        {
          std::cerr << "rank 0: " << failure << '\n';
        }
      }
      else
      {
        std::cout << "PASS: " << sc25::coupled_fixture_id << " ("
                  << coupled_evidence.full_dimension << " unknowns, "
                  << coupled_evidence.checked_coordinate_directions
                  << " coordinate directions, max FD discrepancy "
                  << coupled_evidence.maximum_directional_discrepancy
                  << ")\n";
      }
    }
  }

  mfem::Hypre::Finalize();
  MPI_Finalize();
  return failures == 0 ? 0 : 1;
}

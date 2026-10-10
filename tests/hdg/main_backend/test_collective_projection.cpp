#include "igrlab/compat.hpp"
#include "sim/hdg/main_backend.hpp"

#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

namespace
{

bool AllRanks(const bool local)
{
  const int input = local ? 1 : 0;
  int output = 0;
  MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return output == 1;
}

class RankLocalNonFiniteEulerCoefficient final : public mfem::VectorCoefficient
{
public:
  RankLocalNonFiniteEulerCoefficient(
      const int rank, const mfem::Geometry::Type failing_geometry)
    : mfem::VectorCoefficient(
          dg::hdg::generic::euler_2d_conserved_components),
      rank_(rank), failing_geometry_(failing_geometry)
  {
  }

  void Eval(mfem::Vector& value, mfem::ElementTransformation& transformation,
            const mfem::IntegrationPoint&) override
  {
    value.SetSize(dg::hdg::generic::euler_2d_conserved_components);
    value = 0.0;
    value(0) = rank_ == 0 &&
        transformation.GetGeometryType() == failing_geometry_
        ? std::numeric_limits<double>::quiet_NaN()
        : 1.0;
    value(3) = 2.5;
  }

private:
  int rank_ = -1;
  mfem::Geometry::Type failing_geometry_ = mfem::Geometry::POINT;
};

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int ranks = 0;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool rejected = false;
  if (ranks == 2)
  {
    dg_sim::CaseConfig case_config;
    case_config.case_name = "hdg_collective_projection_rejection";
    case_config.gamma = mfem::igrlab::EulerDefaults::kGamma;
    case_config.case_dir = DG_IGR_HDG_REPOSITORY_ROOT;

    dg_sim::CommonOptions options;
    options.mesh_file = std::string(DG_IGR_HDG_REPOSITORY_ROOT) +
        "/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh";
    options.order = 1;
    options.par_ref_levels = 0;
    options.device = "cpu";
    options.regularization = "none";
    options.dt = 0.1;
    options.dt_min = 0.0;
    options.cfl = 0.25;
    options.t_final = 0.1;
    options.vtk_save = false;

    const auto run_rejection = [&](const mfem::Geometry::Type failing_geometry)
    {
      const dg_sim::hdg::HdgEulerCaseDataFactory make_case_data =
          [rank, failing_geometry, &options](const dg_sim::CaseContext&)
      {
        dg_sim::hdg::HdgEulerCaseData case_data;
        case_data.initial_condition =
            std::make_unique<RankLocalNonFiniteEulerCoefficient>(
                rank, failing_geometry);
        case_data.numerical_policy =
            dg::hdg::generic::DefaultEulerHdgNumericalPolicy(options.order);
        return case_data;
      };
      const dg_sim::hdg::HdgEulerMainRequest request{
          case_config, options, make_case_data, MPI_COMM_WORLD, {}};
      return dg_sim::hdg::RunEulerMainBackend(request) != 0;
    };
    const bool volume_rejected = run_rejection(mfem::Geometry::SQUARE);
    const bool trace_rejected = run_rejection(mfem::Geometry::SEGMENT);
    rejected = volume_rejected && trace_rejected;
  }

  const bool passed = ranks == 2 && AllRanks(rejected);
  if (!passed && rank == 0)
  {
    std::cerr << "HDG collective projection rejection test failed\n";
  }
  MPI_Finalize();
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

#include "sim/hdg/main_backend.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

class SmoothEulerCoefficient final : public mfem::VectorCoefficient
{
public:
  SmoothEulerCoefficient() : mfem::VectorCoefficient(4) {}

  void Eval(mfem::Vector& state, mfem::ElementTransformation& transformation,
            const mfem::IntegrationPoint& integration_point) override
  {
    mfem::Vector point;
    transformation.Transform(integration_point, point);
    const double rho = 1.0 + 0.01 *
        std::sin(2.0 * 3.14159265358979323846 * point[0]) *
        std::sin(2.0 * 3.14159265358979323846 * point[1]);
    constexpr double u = 0.02;
    constexpr double v = -0.01;
    constexpr double pressure = 1.0;
    state.SetSize(4);
    state[0] = rho;
    state[1] = rho * u;
    state[2] = rho * v;
    state[3] = pressure / 0.4 + 0.5 * rho * (u * u + v * v);
  }
};

class SmoothSourceCoefficient final : public mfem::VectorCoefficient
{
public:
  SmoothSourceCoefficient() : mfem::VectorCoefficient(4) {}

  void Eval(mfem::Vector& source, mfem::ElementTransformation& transformation,
            const mfem::IntegrationPoint& integration_point) override
  {
    mfem::Vector point;
    transformation.Transform(integration_point, point);
    const double modulation = 1.0 + 0.1 * point[0] - 0.05 * point[1];
    source.SetSize(4);
    source[0] = 1.0e-3 * modulation;
    source[1] = 2.0e-4 * modulation;
    source[2] = -1.0e-4 * modulation;
    source[3] = 2.5e-3 * modulation;
  }
};

mfem::Vector ReferenceState()
{
  mfem::Vector state(4);
  state[0] = 1.0;
  state[1] = 0.02;
  state[2] = -0.01;
  state[3] = 2.50025;
  return state;
}

mfem::Array<int> Marker(const int maximum_attribute, const int attribute)
{
  if (maximum_attribute != 7 || attribute < 1 ||
      attribute > maximum_attribute)
  {
    throw std::invalid_argument(
        "CUDA physical-boundary qualification mesh must expose attributes 1-7");
  }
  mfem::Array<int> marker(maximum_attribute);
  marker = 0;
  marker[attribute - 1] = 1;
  return marker;
}

dg_sim::hdg::HdgEulerCaseData MakeCaseData(
    const dg_sim::CaseContext& context, const int order)
{
  dg_sim::hdg::HdgEulerCaseData data;
  data.initial_condition = std::make_unique<SmoothEulerCoefficient>();
  data.static_source = std::make_unique<SmoothSourceCoefficient>();
  data.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);

  const int attributes = context.Mesh().bdr_attributes.Size() == 0
      ? 0
      : context.Mesh().bdr_attributes.Max();
  data.boundary_plan.AddSupersonicInlet(
      Marker(attributes, 1), [] { return ReferenceState(); });
  data.boundary_plan.AddSubsonicInlet(
      Marker(attributes, 2), [] { return ReferenceState(); });
  data.boundary_plan.AddSupersonicOutlet(Marker(attributes, 3));
  data.boundary_plan.AddSubsonicOutletPressure(
      Marker(attributes, 4), [] { return 1.0; });
  data.boundary_plan.AddFarField(
      Marker(attributes, 5), [] { return ReferenceState(); });
  data.boundary_plan.AddNonReflectingOutlet(
      Marker(attributes, 6), [] { return ReferenceState(); });
  data.boundary_plan.AddSlipWall(Marker(attributes, 7));
  return data;
}

bool All(const bool local)
{
  const int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS &&
      output == 1;
}

bool RunCase(const int order,
             const dg::hdg::generic::RegularizationKind regularization,
             const dg::hdg::generic::CudaEngineKind cuda_engine)
{
  dg_sim::CaseConfig case_config;
  case_config.case_name = "cuda_physical_extension";
  case_config.gamma = mfem::dg_igr::EulerDefaults::kGamma;
  case_config.case_dir = DG_IGR_HDG_REPOSITORY_ROOT;

  dg_sim::CommonOptions options;
  options.mesh_file = DG_IGR_HDG_CUDA_PHYSICAL_MESH;
  options.ser_ref_levels = 0;
  options.par_ref_levels = 0;
  options.order = order;
  options.device = "cuda";
  options.discretization = "hdg";
  options.hdg_cuda_engine = cuda_engine ==
          dg::hdg::generic::CudaEngineKind::tensor_core
      ? "tensor"
      : "ordinary";
  options.regularization = regularization ==
          dg::hdg::generic::RegularizationKind::igr
      ? "igr"
      : "none";
  options.hdg_igr_alpha = 0.2;
  options.hdg_igr_beta = 0.8;
  options.dt = 1.0e-5;
  options.dt_min = 0.0;
  options.cfl = 0.25;
  options.t_final = 1.0e-5;
  options.vtk_save = false;
  options.glvis = false;
  options.analysis_steps = 0;

  const dg_sim::hdg::HdgEulerCaseDataFactory factory =
      [order](const dg_sim::CaseContext& context)
      { return MakeCaseData(context, order); };
  const dg_sim::hdg::HdgEulerMainRequest request{
      case_config, options, factory, MPI_COMM_WORLD};
  return dg_sim::hdg::RunEulerMainBackend(request) == 0;
}

bool ParseEngine(const int argc, char* argv[],
                 dg::hdg::generic::CudaEngineKind& engine,
                 bool& rank_mismatch)
{
  engine = dg::hdg::generic::CudaEngineKind::ordinary;
  rank_mismatch = false;
  if (argc == 1) return true;
  if (argc == 2 && std::string(argv[1]) == "--rank-engine-mismatch")
  {
    rank_mismatch = true;
    return true;
  }
  if (argc != 3 || std::string(argv[1]) != "--engine") return false;
  const std::string value(argv[2]);
  if (value == "ordinary") return true;
  if (value == "tensor")
  {
    engine = dg::hdg::generic::CudaEngineKind::tensor_core;
    return true;
  }
  return false;
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  int ranks = 0;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  dg::hdg::generic::CudaEngineKind cuda_engine =
      dg::hdg::generic::CudaEngineKind::ordinary;
  bool rank_engine_mismatch = false;
  if (!ParseEngine(argc, argv, cuda_engine, rank_engine_mismatch))
  {
    if (rank == 0)
      std::cerr << "Usage: " << argv[0]
                << " [--engine ordinary|tensor] [--rank-engine-mismatch]\n";
    return 2;
  }
  if (rank_engine_mismatch)
  {
    if (ranks != 2) return 2;
    cuda_engine = rank == 0
        ? dg::hdg::generic::CudaEngineKind::ordinary
        : dg::hdg::generic::CudaEngineKind::tensor_core;
    const bool rejected = !RunCase(
        1, dg::hdg::generic::RegularizationKind::none, cuda_engine);
    return All(rejected) ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  bool conventional = true;
  bool igr = true;
  for (int order = dg::hdg::generic::minimum_polynomial_order;
       order <= dg::hdg::generic::maximum_polynomial_order; ++order)
  {
    // Do not short-circuit collective runs after a prior failure: every rank
    // must enter the complete qualification matrix in the same order.
    const bool conventional_order = RunCase(
        order, dg::hdg::generic::RegularizationKind::none, cuda_engine);
    conventional = conventional && conventional_order;
    const bool igr_order = RunCase(
        order, dg::hdg::generic::RegularizationKind::igr, cuda_engine);
    igr = igr && igr_order;
  }
  const bool passed = (ranks == 1 || ranks == 2) &&
      All(conventional && igr);
  if (!passed)
  {
    std::cerr << "CUDA physical-boundary extension qualification failed rank="
              << rank << " ranks=" << ranks
              << " engine="
              << (cuda_engine == dg::hdg::generic::CudaEngineKind::tensor_core
                      ? "tensor"
                      : "ordinary")
              << " conventional=" << conventional << " igr=" << igr
              << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

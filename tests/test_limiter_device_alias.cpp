#include "igrlab/compat.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "physics/euler/eos.hpp"
#include "utils/diagnostics/euler_admissibility.hpp"

namespace
{

constexpr int kDim = 2;
constexpr int kNumEq = kDim + 2;
constexpr double kGamma = 1.4;

double MinDofPressure(mfem::ParFiniteElementSpace& vfes,
                      const mfem::Vector& state)
{
  const int ndofs = vfes.GetNDofs();
  const double* data = state.HostRead();
  double local_min = std::numeric_limits<double>::infinity();
  for (int d = 0; d < ndofs; ++d)
  {
    mfem::Vector value(kNumEq);
    for (int c = 0; c < kNumEq; ++c)
    {
      value(c) = data[d + ndofs * c];
    }
    local_min = std::min(local_min,
                         mfem::EulerPressureFromConservative(value,
                                                             kDim,
                                                             kGamma,
                                                             1.0e-16));
  }
  double global_min = local_min;
  MPI_Allreduce(MPI_IN_PLACE,
                &global_min,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MIN,
                vfes.GetComm());
  return global_min;
}

class StageOneInadmissibilityOperator final
  : public mfem::TimeDependentOperator
{
public:
  explicit StageOneInadmissibilityOperator(mfem::ParFiniteElementSpace& vfes)
    : mfem::TimeDependentOperator(vfes.GetTrueVSize()), vfes_(vfes)
  {
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    if (calls_ == 1)
    {
      const auto scan = mfem::igrlab::ScanEulerAdmissibility(
        vfes_,
        x,
        kDim,
        kGamma,
        1.0e-10,
        1.0e-10,
        0,
        {},
        mfem::igrlab::EulerAdmissibilityPointSet::Limiter);
      stage_one_bad_pressure_ = scan.global_bad_pressure;
    }

    y.SetSize(height);
    y = 0.0;
    if (calls_ == 0)
    {
      const int ndofs = vfes_.GetNDofs();
      y.HostReadWrite()[3 * ndofs] = -3000.0;
    }
    ++calls_;
  }

  int StageOneBadPressure() const { return stage_one_bad_pressure_; }

private:
  mfem::ParFiniteElementSpace& vfes_;
  mutable int calls_ = 0;
  mutable int stage_one_bad_pressure_ = -1;
};

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const char* device_config = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(device_config);

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mfem::DG_FECollection fec(2, kDim, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace vfes(
    &pmesh, &fec, kNumEq, mfem::Ordering::byNODES);

  mfem::ParGridFunction point_set_initial(&vfes);
  point_set_initial = 0.0;
  const mfem::IntegrationRule& nodes = vfes.GetFE(0)->GetNodes();
  const int point_set_ndofs = vfes.GetNDofs();
  double* point_set_data = point_set_initial.HostReadWrite();
  for (int d = 0; d < point_set_ndofs; ++d)
  {
    const mfem::IntegrationPoint& ip = nodes.IntPoint(d);
    const double pressure = (ip.x - 0.5) * (ip.x - 0.5) +
                            (ip.y - 0.5) * (ip.y - 0.5) - 0.02;
    point_set_data[d] = 1.0;
    point_set_data[d + point_set_ndofs] = 0.0;
    point_set_data[d + 2 * point_set_ndofs] = 0.0;
    point_set_data[d + 3 * point_set_ndofs] = pressure / (kGamma - 1.0);
  }

  mfem::ParGridFunction quadrature_only(&vfes);
  quadrature_only = point_set_initial;
  mfem::ApplyZhangShuPositivityLimiter(
    vfes,
    quadrature_only,
    kDim,
    kGamma,
    1.0e-10,
    1.0e-10,
    6,
    {5},
    16,
    mfem::ZhangShuPointSet::kQuadrature,
    /*diagnostics=*/nullptr,
    /*enforce_vtk_points=*/false);
  if (!(MinDofPressure(vfes, quadrature_only) < 0.0))
  {
    std::cerr
      << "[FAIL] CUDA quadrature policy unexpectedly constrained FE nodes\n";
    return 3;
  }

  mfem::ParGridFunction vtk_points(&vfes);
  vtk_points = point_set_initial;
  mfem::ApplyZhangShuPositivityLimiter(
    vfes,
    vtk_points,
    kDim,
    kGamma,
    1.0e-10,
    1.0e-10,
    6,
    {5},
    16,
    mfem::ZhangShuPointSet::kQuadrature,
    /*diagnostics=*/nullptr,
    /*enforce_vtk_points=*/true);
  if (!(MinDofPressure(vfes, vtk_points) >= 0.99e-10))
  {
    std::cerr << "[FAIL] CUDA VTK-point policy did not constrain refined "
                 "output points\n";
    return 7;
  }

  mfem::ParGridFunction consumer(&vfes);
  consumer = point_set_initial;
  mfem::ApplyZhangShuPositivityLimiter(
    vfes,
    consumer,
    kDim,
    kGamma,
    1.0e-10,
    1.0e-10,
    6,
    {5},
    16,
    mfem::ZhangShuPointSet::kConsumer);
  if (!(MinDofPressure(vfes, consumer) >= 0.99e-10))
  {
    std::cerr << "[FAIL] CUDA consumer policy did not constrain FE nodes\n";
    return 4;
  }

  mfem::ParGridFunction initial(&vfes);
  initial = 0.0;
  const int ndofs = vfes.GetNDofs();
  double* data = initial.HostReadWrite();
  for (int d = 0; d < ndofs; ++d)
  {
    data[d] = 1.0;
    data[d + ndofs] = 0.0;
    data[d + 2 * ndofs] = 0.0;
    data[d + 3 * ndofs] = 2.5;
  }
  const double p_min = 1.0e-10;
  const double pressure_before = MinDofPressure(vfes, initial);
  if (!(pressure_before > p_min))
  {
    std::cerr << "[FAIL] test initial state is not pressure-admissible\n";
    return 2;
  }

  int global_ndofs = ndofs;
  MPI_Allreduce(MPI_IN_PLACE,
                &global_ndofs,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_SUM,
                vfes.GetComm());
  bool nonfinite_scan_ok = true;
  for (int component = 0; component < kNumEq; ++component)
  {
    mfem::ParGridFunction nonfinite_state(&vfes);
    nonfinite_state = initial;
    double* nonfinite_data = nonfinite_state.HostReadWrite();
    for (int d = 0; d < ndofs; ++d)
    {
      nonfinite_data[d + component * ndofs] =
        std::numeric_limits<double>::quiet_NaN();
    }
    const auto scan = mfem::igrlab::ScanEulerAdmissibility(
      vfes,
      nonfinite_state,
      kDim,
      kGamma,
      1.0e-10,
      p_min,
      0,
      {},
      mfem::igrlab::EulerAdmissibilityPointSet::Dof);
    nonfinite_scan_ok = nonfinite_scan_ok &&
                        scan.global_bad_pressure == global_ndofs &&
                        scan.global_bad_rho ==
                          (component == 0 ? global_ndofs : 0);
  }

  mfem::ParGridFunction nonfinite_pressure(&vfes);
  nonfinite_pressure = initial;
  double* nonfinite_pressure_data = nonfinite_pressure.HostReadWrite();
  for (int d = 0; d < ndofs; ++d)
  {
    nonfinite_pressure_data[d + 3 * ndofs] =
      std::numeric_limits<double>::max();
  }
  const auto nonfinite_pressure_scan =
    mfem::igrlab::ScanEulerAdmissibility(
      vfes,
      nonfinite_pressure,
      kDim,
      /*gamma=*/3.0,
      1.0e-10,
      p_min,
      0,
      {},
      mfem::igrlab::EulerAdmissibilityPointSet::Dof);
  nonfinite_scan_ok = nonfinite_scan_ok &&
                      nonfinite_pressure_scan.global_bad_rho == 0 &&
                      nonfinite_pressure_scan.global_bad_pressure ==
                        global_ndofs;
  if (!nonfinite_scan_ok)
  {
    std::cerr << "[FAIL] admissibility scan accepted a nonfinite state or "
                 "derived pressure\n";
    return 6;
  }

  mfem::ParGridFunction invalid_mean(&vfes);
  invalid_mean = initial;
  double* invalid_data = invalid_mean.HostReadWrite();
  for (int d = 0; d < ndofs; ++d)
  {
    invalid_data[d] = -1.0;
  }
  const mfem::Vector invalid_before(invalid_mean);
  int rejected_elements = 0;
  try
  {
    mfem::ApplyZhangShuPositivityLimiter(
      vfes,
      invalid_mean,
      kDim,
      kGamma,
      1.0e-10,
      p_min,
      6,
      {5},
      16,
      mfem::ZhangShuPointSet::kConsumer);
  }
  catch (const mfem::RecoverableLimiterFailure& failure)
  {
    rejected_elements = failure.GlobalRejectedElementCount();
  }
  mfem::Vector invalid_error(invalid_mean);
  invalid_error -= invalid_before;
  if (rejected_elements <= 0 || invalid_error.Normlinf() != 0.0)
  {
    std::cerr << "[FAIL] device limiter repaired or committed an invalid mean\n";
    return 5;
  }

  mfem::Vector x;
  initial.GetTrueDofs(x);
  StageOneInadmissibilityOperator op(vfes);
  mfem::RK3SSPWithLimiter rk(vfes);
  rk.SetEulerParams(kDim, kGamma);
  rk.SetLimiterParams(1.0e-10, p_min, 0, 16);
  rk.LimitInitial(true);
  rk.AddZhangShuLimiter(mfem::ZhangShuPointSet::kConsumer,
                        /*always_on=*/true);
  rk.EnableStageAdmissibilityScan(true);
  rk.Init(op);

  double t = 0.0;
  double dt = 1.0e-3;
  rk.Step(x, t, dt);

  if (op.StageOneBadPressure() != 0)
  {
    std::cerr << "[FAIL] next RHS observed stale pressure-inadmissible stage"
              << " bad_pressure=" << op.StageOneBadPressure() << "\n";
    return 1;
  }

  if (mfem::Mpi::Root())
  {
    std::cout << "[PASS] CUDA consumer point policy constrains FE nodes\n";
    std::cout << "[PASS] CUDA typed VTK-point policy constrains output points\n";
    std::cout << "[PASS] admissibility scan rejects nonfinite states and "
                 "derived pressure\n";
    std::cout << "[PASS] CUDA invalid mean rejection is transactional\n";
    std::cout << "[PASS] next RHS observes the limited device alias\n";
  }
  return 0;
}

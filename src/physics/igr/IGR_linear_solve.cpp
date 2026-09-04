#include "physics/igr/IGR.hpp"

#include "numerics/timestepping/ssprk_coefficients.hpp"
#include "utils/config/env_flags.hpp"
#include "utils/profiling/nvtx.hpp"
#include "IGR_device_kernels.hpp"
#include "IGR_internal_utils.hpp"
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dg_igr
{
using namespace igr_internal;

namespace
{

using Clock = std::chrono::steady_clock;

double ElapsedSeconds(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct ExactFaceTimingStats
{
  long long mult_calls = 0;
  double base_seconds = 0.0;
  double face_seconds = 0.0;
  double sum_axpy_seconds = 0.0;
  double face_setup_seconds = 0.0;
  double prolong_seconds = 0.0;
  double local_apply_seconds = 0.0;
  double shared_exchange_seconds = 0.0;
  double shared_apply_seconds = 0.0;
  double restrict_seconds = 0.0;
};

int ReadEnvInt(const char* name, int default_value)
{
  const char* raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
  {
    return default_value;
  }
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  MFEM_VERIFY(end && *end == '\0',
              "Invalid integer value for " << name << ": " << raw);
  MFEM_VERIFY(value >= std::numeric_limits<int>::min() &&
                value <= std::numeric_limits<int>::max(),
              "Integer value for " << name << " is out of range: " << raw);
  return static_cast<int>(value);
}

mfem::real_t ReadEnvReal(const char* name, mfem::real_t default_value)
{
  const char* raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
  {
    return default_value;
  }
  char* end = nullptr;
  const double value = std::strtod(raw, &end);
  MFEM_VERIFY(end && *end == '\0',
              "Invalid real value for " << name << ": " << raw);
  return static_cast<mfem::real_t>(value);
}

dg_face::DGDiffusionFaceScaling ReadIPDGFaceScaling()
{
  const char* raw = std::getenv("DG_IGR_IGR_IPDG_FACE_SCALING");
  const std::string value = raw ? raw : "arithmetic";
  if (value == "arithmetic")
  {
    return dg_face::DGDiffusionFaceScaling::Arithmetic;
  }
  if (value == "max")
  {
    return dg_face::DGDiffusionFaceScaling::Max;
  }
  if (value == "sum")
  {
    return dg_face::DGDiffusionFaceScaling::Sum;
  }
  if (value == "harmonic")
  {
    return dg_face::DGDiffusionFaceScaling::HarmonicWeighted;
  }
  MFEM_ABORT("DG_IGR_IGR_IPDG_FACE_SCALING must be one of "
             "arithmetic|max|sum|harmonic.");
  return dg_face::DGDiffusionFaceScaling::Arithmetic;
}

const char* IPDGFaceScalingName(dg_face::DGDiffusionFaceScaling value)
{
  switch (value)
  {
  case dg_face::DGDiffusionFaceScaling::Arithmetic:
    return "arithmetic";
  case dg_face::DGDiffusionFaceScaling::Max:
    return "max";
  case dg_face::DGDiffusionFaceScaling::Sum:
    return "sum";
  case dg_face::DGDiffusionFaceScaling::HarmonicWeighted:
    return "harmonic";
  }
  return "arithmetic";
}

double OperatorProbeDot(const mfem::Vector& x,
                        const mfem::Vector& y,
                        MPI_Comm comm)
{
  const double local = x * y;
  double global = local;
#ifdef MFEM_USE_MPI
  MPI_Allreduce(&local,
                &global,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM,
                comm);
#else
  (void)comm;
#endif
  return global;
}

double OperatorProbeNorm(const mfem::Vector& x, MPI_Comm comm)
{
  return std::sqrt(std::max(OperatorProbeDot(x, x, comm), 0.0));
}

void WriteVectorLines(const std::filesystem::path& path,
                      const mfem::Vector& vector)
{
  std::ofstream output(path);
  output << std::setprecision(17);
  const double* data = vector.HostRead();
  for (int i = 0; i < vector.Size(); ++i)
  {
    output << data[i] << '\n';
  }
}

void SaveLinearEventCapture(
  const std::filesystem::path& prefix,
  int solve_call,
  int rk_step,
  int rk_stage,
  double time,
  double dt,
  const mfem::ParGridFunction& state,
  const mfem::ParFiniteElementSpace& sigma_fes,
  const mfem::Vector& rhs,
  const mfem::Vector& initial_guess,
  const mfem::Vector& solution,
  int iterations,
  bool converged,
  double reported_initial_norm,
  double reported_final_norm,
  double true_initial_residual_abs,
  double true_residual_abs,
  double true_residual_rel_rhs,
  double true_residual_rel_initial,
  double recomputed_preconditioned_initial_norm,
  double recomputed_preconditioned_final_norm,
  double effective_rtol,
  bool warm_start_requested,
  bool warm_start_effective,
  const std::string& solver,
  const std::string& preconditioner,
  double chebyshev_eigenvalue)
{
  const int rank = mfem::Mpi::WorldRank();
  std::ostringstream rank_name;
  rank_name << std::setfill('0') << std::setw(6) << rank;
  const std::filesystem::path output_dir =
    prefix / ("solve-" + std::to_string(solve_call));
  std::filesystem::create_directories(output_dir);

  {
    std::ofstream output(output_dir /
                         ("euler-mesh-final." + rank_name.str()));
    output.precision(17);
    output << *state.ParFESpace()->GetParMesh();
  }

  const auto* state_fes = state.ParFESpace();
  MFEM_VERIFY(state_fes->GetOrdering() == mfem::Ordering::byNODES,
              "Linear-event state capture requires byNODES ordering.");
  const int ndofs = state_fes->GetNDofs();
  const double* state_data = state.HostRead();
  mfem::ParFiniteElementSpace scalar_fes(state_fes->GetParMesh(),
                                         state_fes->FEColl(),
                                         1,
                                         mfem::Ordering::byNODES);
  for (int component = 0; component < state_fes->GetVDim(); ++component)
  {
    mfem::ParGridFunction field(&scalar_fes,
                                const_cast<double*>(state_data) +
                                  component * ndofs);
    std::ofstream output(
      output_dir /
      ("euler-" + std::to_string(component) + "-final." + rank_name.str()));
    output.precision(17);
    output << field;
  }

  WriteVectorLines(output_dir / ("rhs." + rank_name.str()), rhs);
  WriteVectorLines(output_dir / ("initial-guess." + rank_name.str()),
                   initial_guess);
  WriteVectorLines(output_dir / ("solution." + rank_name.str()), solution);

  if (mfem::Mpi::Root())
  {
    std::ofstream output(output_dir / "metadata.txt");
    output << std::setprecision(17)
           << "solve_call=" << solve_call << '\n'
           << "rk_step=" << rk_step << '\n'
           << "rk_stage=" << rk_stage << '\n'
           << "time=" << time << '\n'
           << "dt=" << dt << '\n'
           << "iterations=" << iterations << '\n'
           << "converged=" << (converged ? 1 : 0) << '\n'
           << "reported_initial_norm=" << reported_initial_norm << '\n'
           << "reported_final_norm=" << reported_final_norm << '\n'
           << "true_initial_residual_abs=" << true_initial_residual_abs << '\n'
           << "true_residual_abs=" << true_residual_abs << '\n'
           << "true_residual_rel_rhs=" << true_residual_rel_rhs << '\n'
           << "true_residual_rel_initial=" << true_residual_rel_initial << '\n'
           << "recomputed_preconditioned_initial_norm="
           << recomputed_preconditioned_initial_norm << '\n'
           << "recomputed_preconditioned_final_norm="
           << recomputed_preconditioned_final_norm << '\n'
           << "effective_rtol=" << effective_rtol << '\n'
           << "warm_start_requested=" << (warm_start_requested ? 1 : 0)
           << '\n'
           << "warm_start_effective=" << (warm_start_effective ? 1 : 0)
           << '\n'
           << "solver=" << solver << '\n'
           << "preconditioner=" << preconditioner << '\n'
           << "chebyshev_eigenvalue=" << chebyshev_eigenvalue << '\n'
           << "rhs_size=" << rhs.Size() << '\n'
           << "sigma_true_size=" << sigma_fes.GetTrueVSize() << '\n';
  }
}

void LogFaceCoefficientAudit(const char* domain,
                             const mfem::Vector& coefficient,
                             double alpha,
                             double rho_floor,
                             int solve_call,
                             MPI_Comm comm,
                             const mfem::Vector* side_alpha = nullptr,
                             int face_quadrature_points = 0)
{
  const double* data = coefficient.HostRead();
  const double* side_alpha_data =
    side_alpha && side_alpha->Size() > 0 ? side_alpha->HostRead() : nullptr;
  MFEM_VERIFY(!side_alpha_data ||
                (face_quadrature_points > 0 &&
                 coefficient.Size() ==
                   side_alpha->Size() * face_quadrature_points),
              "Face-coefficient audit received inconsistent local-alpha data.");
  double local_min = std::numeric_limits<double>::infinity();
  double local_max = -std::numeric_limits<double>::infinity();
  long long local_values = coefficient.Size();
  long long local_floor = 0;
  long long local_nonfinite = 0;
  for (int i = 0; i < coefficient.Size(); ++i)
  {
    const int side = i % 2;
    const int face = side_alpha_data
      ? i / (2 * face_quadrature_points)
      : 0;
    const double coefficient_alpha = side_alpha_data
      ? side_alpha_data[side + 2 * face]
      : alpha;
    const double floor_value = coefficient_alpha / rho_floor;
    local_min = std::min(local_min, data[i]);
    local_max = std::max(local_max, data[i]);
    local_floor += data[i] >= floor_value * (1.0 - 1.0e-12) ? 1 : 0;
    local_nonfinite += std::isfinite(data[i]) ? 0 : 1;
  }
  double global_min = local_min;
  double global_max = local_max;
  long long global_values = local_values;
  long long global_floor = local_floor;
  long long global_nonfinite = local_nonfinite;
  MPI_Allreduce(MPI_IN_PLACE, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(MPI_IN_PLACE,
                &global_values,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                comm);
  MPI_Allreduce(MPI_IN_PLACE,
                &global_floor,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                comm);
  MPI_Allreduce(MPI_IN_PLACE,
                &global_nonfinite,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                comm);
  if (mfem::Mpi::Root())
  {
    std::cout << "[igr-ingestion-audit] kind=face-coefficient"
              << " solve=" << solve_call
              << " domain=" << domain
              << " values=" << global_values
              << " min=" << global_min
              << " max=" << global_max
              << " density_floor_activations=" << global_floor
              << " nonfinite=" << global_nonfinite << std::endl;
  }
}

void BuildSharedFaceSideAlpha(const mfem::ParFiniteElementSpace& fes,
                              mfem::real_t c_alpha,
                              mfem::Vector& face_side_alpha)
{
#ifndef MFEM_USE_MPI
  (void)fes;
  (void)c_alpha;
  (void)face_side_alpha;
  MFEM_ABORT("Shared local IGR alpha lowering requires MFEM MPI support.");
#else
  mfem::ParMesh* pmesh = fes.GetParMesh();
  MFEM_VERIFY(pmesh != nullptr,
              "Shared local IGR alpha lowering requires ParMesh.");
  MFEM_VERIFY(std::isfinite(c_alpha) && c_alpha > 0.0,
              "Shared local IGR alpha lowering requires positive finite "
              "C_alpha.");
  const int nf = pmesh->GetNSharedFaces();
  face_side_alpha.SetSize(2 * nf);
  auto alpha = mfem::Reshape(face_side_alpha.HostWrite(), 2, nf);
  for (int sf = 0; sf < nf; ++sf)
  {
    mfem::FaceElementTransformations* Tr =
      pmesh->GetSharedFaceTransformations(sf, /*fill2=*/true);
    MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                "Shared local IGR alpha lowering requires two elements.");
    const int nbr_el = Tr->Elem2No - pmesh->GetNE();
    MFEM_VERIFY(Tr->Elem1No >= 0 && Tr->Elem1No < pmesh->GetNE() &&
                  nbr_el >= 0 &&
                  nbr_el < pmesh->GetNFaceNeighborElements(),
                "Shared local IGR alpha lowering received an invalid element "
                "pair.");
    const mfem::real_t h0 = pmesh->GetElementSize(Tr->Elem1No, 1);
    const mfem::real_t h1 = pmesh->GetFaceNbrElementSize(nbr_el, 1);
    MFEM_VERIFY(std::isfinite(h0) && std::isfinite(h1) && h0 > 0.0 &&
                  h1 > 0.0,
                "Shared local IGR alpha lowering received an invalid element "
                "size.");
    alpha(0, sf) = c_alpha * h0 * h0;
    alpha(1, sf) = c_alpha * h1 * h1;
  }
#endif
}

void FillOperatorProbeVector(mfem::Vector& value, int seed)
{
  double* data = value.HostWrite();
  const int rank = mfem::Mpi::WorldRank();
  for (int i = 0; i < value.Size(); ++i)
  {
    const double index = static_cast<double>(i + 1 + 104729 * rank);
    data[i] = std::sin((0.013 + 0.001 * seed) * index) +
              0.5 * std::cos((0.021 + 0.002 * seed) * index);
  }
}

struct OperatorAlgebraProbe
{
  double symmetry_defect = std::numeric_limits<double>::quiet_NaN();
  double repeatability_defect = std::numeric_limits<double>::quiet_NaN();
  double min_rayleigh = std::numeric_limits<double>::quiet_NaN();
  double min_ritz = std::numeric_limits<double>::quiet_NaN();
  double max_ritz = std::numeric_limits<double>::quiet_NaN();
  int lanczos_steps = 0;
};

void SymmetricEigenvalueBounds(mfem::DenseMatrix matrix,
                               double& min_eigenvalue,
                               double& max_eigenvalue)
{
  const int n = matrix.Height();
  MFEM_VERIFY(n > 0 && matrix.Width() == n,
              "Symmetric eigenvalue probe requires a square matrix.");
  for (int sweep = 0; sweep < 50 * n * n; ++sweep)
  {
    int p = 0;
    int q = 0;
    double largest = 0.0;
    double diagonal_scale = 1.0;
    for (int i = 0; i < n; ++i)
    {
      diagonal_scale = std::max(diagonal_scale, std::abs(matrix(i, i)));
      for (int j = i + 1; j < n; ++j)
      {
        if (std::abs(matrix(i, j)) > largest)
        {
          largest = std::abs(matrix(i, j));
          p = i;
          q = j;
        }
      }
    }
    if (largest <= 1.0e-13 * diagonal_scale)
    {
      break;
    }

    const double angle =
      0.5 * std::atan2(2.0 * matrix(p, q), matrix(q, q) - matrix(p, p));
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double app = matrix(p, p);
    const double aqq = matrix(q, q);
    const double apq = matrix(p, q);
    matrix(p, p) = cosine * cosine * app - 2.0 * sine * cosine * apq +
                   sine * sine * aqq;
    matrix(q, q) = sine * sine * app + 2.0 * sine * cosine * apq +
                   cosine * cosine * aqq;
    matrix(p, q) = 0.0;
    matrix(q, p) = 0.0;
    for (int k = 0; k < n; ++k)
    {
      if (k == p || k == q)
      {
        continue;
      }
      const double akp = matrix(k, p);
      const double akq = matrix(k, q);
      matrix(k, p) = cosine * akp - sine * akq;
      matrix(p, k) = matrix(k, p);
      matrix(k, q) = sine * akp + cosine * akq;
      matrix(q, k) = matrix(k, q);
    }
  }

  min_eigenvalue = matrix(0, 0);
  max_eigenvalue = matrix(0, 0);
  for (int i = 1; i < n; ++i)
  {
    min_eigenvalue = std::min(min_eigenvalue, matrix(i, i));
    max_eigenvalue = std::max(max_eigenvalue, matrix(i, i));
  }
}

OperatorAlgebraProbe ProbeOperatorAlgebra(const mfem::Operator& op,
                                          MPI_Comm comm,
                                          int requested_steps)
{
  OperatorAlgebraProbe result;
  const int n = op.Width();
  MFEM_VERIFY(n > 0 && op.Height() == n,
              "IGR operator algebra probe requires a non-empty square operator.");

  mfem::Vector x(n), y(n), Ax(n), Ax_repeat(n), Ay(n);
  FillOperatorProbeVector(x, 1);
  FillOperatorProbeVector(y, 2);
  op.Mult(x, Ax);
  op.Mult(x, Ax_repeat);
  op.Mult(y, Ay);
  Ax_repeat -= Ax;
  result.repeatability_defect =
    std::sqrt(std::max(OperatorProbeDot(Ax_repeat, Ax_repeat, comm), 0.0)) /
    std::max(std::sqrt(std::max(OperatorProbeDot(Ax, Ax, comm), 0.0)), 1.0e-30);
  const double xAy = OperatorProbeDot(x, Ay, comm);
  const double yAx = OperatorProbeDot(y, Ax, comm);
  result.symmetry_defect =
    std::abs(xAy - yAx) / std::max({1.0, std::abs(xAy), std::abs(yAx)});
  const double x2 = std::max(OperatorProbeDot(x, x, comm), 1.0e-30);
  const double y2 = std::max(OperatorProbeDot(y, y, comm), 1.0e-30);
  result.min_rayleigh =
    std::min(OperatorProbeDot(x, Ax, comm) / x2,
             OperatorProbeDot(y, Ay, comm) / y2);

  const int max_steps = std::max(1, std::min(requested_steps, n));
  mfem::Vector q(n), previous(n), work(n);
  FillOperatorProbeVector(q, 7);
  q *= 1.0 / std::sqrt(std::max(OperatorProbeDot(q, q, comm), 1.0e-30));
  previous = 0.0;
  std::vector<mfem::Vector> basis;
  std::vector<double> alpha;
  std::vector<double> beta;
  basis.reserve(max_steps);
  alpha.reserve(max_steps);
  beta.reserve(max_steps);
  double previous_beta = 0.0;

  for (int step = 0; step < max_steps; ++step)
  {
    basis.emplace_back(q);
    op.Mult(q, work);
    const double diagonal = OperatorProbeDot(q, work, comm);
    alpha.push_back(diagonal);
    work.Add(-diagonal, q);
    if (step > 0)
    {
      work.Add(-previous_beta, previous);
    }
    for (const mfem::Vector& vector : basis)
    {
      work.Add(-OperatorProbeDot(vector, work, comm), vector);
    }
    const double next_beta =
      std::sqrt(std::max(OperatorProbeDot(work, work, comm), 0.0));
    if (step + 1 >= max_steps || next_beta <= 1.0e-14)
    {
      break;
    }
    beta.push_back(next_beta);
    previous = q;
    q = work;
    q *= 1.0 / next_beta;
    previous_beta = next_beta;
  }

  result.lanczos_steps = static_cast<int>(alpha.size());
  mfem::DenseMatrix tridiagonal(result.lanczos_steps);
  tridiagonal = 0.0;
  for (int i = 0; i < result.lanczos_steps; ++i)
  {
    tridiagonal(i, i) = alpha[i];
    if (i + 1 < result.lanczos_steps)
    {
      tridiagonal(i, i + 1) = beta[i];
      tridiagonal(i + 1, i) = beta[i];
    }
  }
  SymmetricEigenvalueBounds(tridiagonal, result.min_ritz, result.max_ritz);
  return result;
}

void LogOperatorAlgebraProbe(const char* component,
                             int solve_call,
                             const mfem::Operator& op,
                             MPI_Comm comm,
                             int lanczos_steps)
{
  const OperatorAlgebraProbe probe =
    ProbeOperatorAlgebra(op, comm, lanczos_steps);
  if (mfem::Mpi::Root())
  {
    std::cout << "[igr-operator-probe] solve_call=" << solve_call
              << " component=" << component
              << " symmetry_defect=" << probe.symmetry_defect
              << " repeatability_defect=" << probe.repeatability_defect
              << " min_rayleigh=" << probe.min_rayleigh
              << " min_ritz=" << probe.min_ritz
              << " max_ritz=" << probe.max_ritz
              << " lanczos_steps=" << probe.lanczos_steps << std::endl;
  }
}

void MakeDiagonalPositive(mfem::Vector& diag)
{
  diag.UseDevice(true);
  diag.Abs();
}

std::string LogToken(std::string value)
{
  for (char& c : value)
  {
    if (c == ' ' || c == '\t' || c == '\n' || c == '=')
    {
      c = '_';
    }
  }
  return value;
}

std::unique_ptr<mfem::OperatorJacobiSmoother>
MakePositiveOperatorJacobi(const mfem::BilinearForm& a,
                           const mfem::Array<int>& ess_tdof_list)
{
  mfem::Vector diag(a.FESpace()->GetTrueVSize());
  a.AssembleDiagonal(diag);
  MakeDiagonalPositive(diag);
  return std::make_unique<mfem::OperatorJacobiSmoother>(diag, ess_tdof_list);
}

std::unique_ptr<mfem::OperatorJacobiSmoother>
MakePositiveOperatorJacobi(const mfem::Operator& op,
                           const mfem::Array<int>& ess_tdof_list)
{
  mfem::Vector diag(op.Height());
  op.AssembleDiagonal(diag);
  MakeDiagonalPositive(diag);
  return std::make_unique<mfem::OperatorJacobiSmoother>(diag, ess_tdof_list);
}

struct KrylovApplyTelemetry
{
  long long operator_apply_count = 0;
  long long preconditioner_apply_count = 0;
  long long preconditioner_operator_apply_count = 0;
  double operator_apply_seconds = 0.0;
  double preconditioner_apply_seconds = 0.0;
  double preconditioner_operator_apply_seconds = 0.0;
  bool record_seconds = false;
  bool sync = false;
  bool inside_preconditioner = false;

  void ResetCounters()
  {
    operator_apply_count = 0;
    preconditioner_apply_count = 0;
    preconditioner_operator_apply_count = 0;
    operator_apply_seconds = 0.0;
    preconditioner_apply_seconds = 0.0;
    preconditioner_operator_apply_seconds = 0.0;
    inside_preconditioner = false;
  }
};

class KrylovTelemetryOperator final : public mfem::Operator
{
public:
  KrylovTelemetryOperator(const mfem::Operator& op,
                          KrylovApplyTelemetry& telemetry,
                          const mfem::Operator* base_component = nullptr,
                          const mfem::Operator* face_component = nullptr,
                          const mfem::ParGridFunction* state = nullptr,
                          mfem::ParFiniteElementSpace* sigma_fes = nullptr,
                          bool* capture_written = nullptr,
                          MPI_Comm comm = MPI_COMM_WORLD,
                          int solve_call = 0)
    : mfem::Operator(op.Height(), op.Width()),
      op_(op),
      telemetry_(telemetry),
      base_component_(base_component),
      face_component_(face_component),
      state_(state),
      sigma_fes_(sigma_fes),
      capture_written_(capture_written),
      comm_(comm),
      solve_call_(solve_call),
      capture_negative_quadratic_(
        base_component != nullptr && face_component != nullptr &&
        ::dg_igr::EnvFlagEnabledDefault(
          "DG_IGR_IGR_CAPTURE_NEGATIVE_QUADRATIC", false))
  {
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/krylov-operator-mult");
    if (!telemetry_.record_seconds)
    {
      op_.Mult(x, y);
      MaybeReportQuadratic(x, y);
      ++telemetry_.operator_apply_count;
      if (telemetry_.inside_preconditioner)
      {
        ++telemetry_.preconditioner_operator_apply_count;
      }
      return;
    }
    const auto start = Clock::now();
    op_.Mult(x, y);
    MaybeReportQuadratic(x, y);
    if (telemetry_.sync)
    {
      MFEM_DEVICE_SYNC;
    }
    const auto stop = Clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    ++telemetry_.operator_apply_count;
    telemetry_.operator_apply_seconds += seconds;
    if (telemetry_.inside_preconditioner)
    {
      ++telemetry_.preconditioner_operator_apply_count;
      telemetry_.preconditioner_operator_apply_seconds += seconds;
    }
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/krylov-operator-mult-transpose");
    if (!telemetry_.record_seconds)
    {
      op_.MultTranspose(x, y);
      ++telemetry_.operator_apply_count;
      if (telemetry_.inside_preconditioner)
      {
        ++telemetry_.preconditioner_operator_apply_count;
      }
      return;
    }
    const auto start = Clock::now();
    op_.MultTranspose(x, y);
    if (telemetry_.sync)
    {
      MFEM_DEVICE_SYNC;
    }
    const auto stop = Clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    ++telemetry_.operator_apply_count;
    telemetry_.operator_apply_seconds += seconds;
    if (telemetry_.inside_preconditioner)
    {
      ++telemetry_.preconditioner_operator_apply_count;
      telemetry_.preconditioner_operator_apply_seconds += seconds;
    }
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return op_.GetMemoryClass();
  }

private:
  void MaybeReportQuadratic(const mfem::Vector& x,
                            const mfem::Vector& y) const
  {
    if (!capture_negative_quadratic_ || negative_quadratic_reported_ ||
        telemetry_.inside_preconditioner)
    {
      return;
    }
    const double full = OperatorProbeDot(x, y, comm_);
    if (full > 0.0)
    {
      return;
    }
    component_work_.SetSize(y.Size(), y);
    base_component_->Mult(x, component_work_);
    const double base = OperatorProbeDot(x, component_work_, comm_);
    face_component_->Mult(x, component_work_);
    const double face = OperatorProbeDot(x, component_work_, comm_);
    const double norm_sq = std::max(OperatorProbeDot(x, x, comm_), 1.0e-300);
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-negative-quadratic] solve_call=" << solve_call_
                << " operator_apply=" << telemetry_.operator_apply_count + 1
                << " full=" << full
                << " base=" << base
                << " face=" << face
                << " base_plus_face=" << base + face
                << " direction_norm_sq=" << norm_sq
                << " full_rayleigh=" << full / norm_sq << std::endl;
    }
    MaybeSaveNegativeDirection(x, full, base, face, norm_sq);
    negative_quadratic_reported_ = true;
  }

  void MaybeSaveNegativeDirection(const mfem::Vector& x,
                                  double full,
                                  double base,
                                  double face,
                                  double norm_sq) const
  {
    const char* prefix = std::getenv("DG_IGR_IGR_NEGATIVE_CAPTURE_PREFIX");
    if (!prefix || prefix[0] == '\0' || !state_ || !sigma_fes_ ||
        !capture_written_ || *capture_written_)
    {
      return;
    }

    const int rank = mfem::Mpi::WorldRank();
    std::ostringstream rank_name;
    rank_name << std::setfill('0') << std::setw(6) << rank;
    const std::filesystem::path output_dir =
      std::filesystem::path(prefix) /
      ("solve-" + std::to_string(solve_call_));
    std::filesystem::create_directories(output_dir);

    {
      std::ofstream output(output_dir /
                           ("euler-mesh-final." + rank_name.str()));
      output.precision(17);
      output << *state_->ParFESpace()->GetParMesh();
    }

    const auto* state_fes = state_->ParFESpace();
    MFEM_VERIFY(state_fes->GetOrdering() == mfem::Ordering::byNODES,
                "Negative-direction state capture requires byNODES ordering.");
    const int ndofs = state_fes->GetNDofs();
    const double* state_data = state_->HostRead();
    mfem::ParFiniteElementSpace scalar_fes(state_fes->GetParMesh(),
                                           state_fes->FEColl(),
                                           1,
                                           mfem::Ordering::byNODES);
    for (int component = 0; component < state_fes->GetVDim(); ++component)
    {
      mfem::ParGridFunction field(&scalar_fes,
                                  const_cast<double*>(state_data) +
                                    component * ndofs);
      std::ofstream output(
        output_dir /
        ("euler-" + std::to_string(component) + "-final." + rank_name.str()));
      output.precision(17);
      output << field;
    }

    {
      std::ofstream output(output_dir /
                           ("direction." + rank_name.str()));
      output << std::setprecision(17);
      const double* direction_data = x.HostRead();
      for (int i = 0; i < x.Size(); ++i)
      {
        output << direction_data[i] << '\n';
      }
    }
    if (mfem::Mpi::Root())
    {
      std::ofstream output(output_dir / "metadata.txt");
      output << std::setprecision(17)
             << "solve_call=" << solve_call_ << '\n'
             << "full=" << full << '\n'
             << "base=" << base << '\n'
             << "face=" << face << '\n'
             << "direction_norm_sq=" << norm_sq << '\n'
             << "direction_size=" << x.Size() << '\n';
      std::cout << "[igr-negative-quadratic] capture_dir="
                << output_dir.string() << std::endl;
    }
    *capture_written_ = true;
  }

  const mfem::Operator& op_;
  KrylovApplyTelemetry& telemetry_;
  const mfem::Operator* base_component_ = nullptr;
  const mfem::Operator* face_component_ = nullptr;
  const mfem::ParGridFunction* state_ = nullptr;
  mfem::ParFiniteElementSpace* sigma_fes_ = nullptr;
  bool* capture_written_ = nullptr;
  MPI_Comm comm_ = MPI_COMM_WORLD;
  int solve_call_ = 0;
  bool capture_negative_quadratic_ = false;
  mutable bool negative_quadratic_reported_ = false;
  mutable mfem::Vector component_work_;
};

class KrylovTelemetryPreconditioner final : public mfem::Solver
{
public:
  KrylovTelemetryPreconditioner(mfem::Solver& preconditioner,
                                KrylovApplyTelemetry& telemetry)
    : mfem::Solver(preconditioner.Height(), preconditioner.Width(), false),
      preconditioner_(preconditioner),
      telemetry_(telemetry)
  {
  }

  void SetOperator(const mfem::Operator&) override
  {
    height = preconditioner_.Height();
    width = preconditioner_.Width();
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/krylov-preconditioner-mult");
    if (!telemetry_.record_seconds)
    {
      const bool was_inside = telemetry_.inside_preconditioner;
      telemetry_.inside_preconditioner = true;
      preconditioner_.Mult(x, y);
      telemetry_.inside_preconditioner = was_inside;
      ++telemetry_.preconditioner_apply_count;
      return;
    }
    const auto start = Clock::now();
    const bool was_inside = telemetry_.inside_preconditioner;
    telemetry_.inside_preconditioner = true;
    preconditioner_.Mult(x, y);
    telemetry_.inside_preconditioner = was_inside;
    if (telemetry_.sync)
    {
      MFEM_DEVICE_SYNC;
    }
    const auto stop = Clock::now();
    ++telemetry_.preconditioner_apply_count;
    telemetry_.preconditioner_apply_seconds +=
      std::chrono::duration<double>(stop - start).count();
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/krylov-preconditioner-mult-transpose");
    if (!telemetry_.record_seconds)
    {
      const bool was_inside = telemetry_.inside_preconditioner;
      telemetry_.inside_preconditioner = true;
      preconditioner_.MultTranspose(x, y);
      telemetry_.inside_preconditioner = was_inside;
      ++telemetry_.preconditioner_apply_count;
      return;
    }
    const auto start = Clock::now();
    const bool was_inside = telemetry_.inside_preconditioner;
    telemetry_.inside_preconditioner = true;
    preconditioner_.MultTranspose(x, y);
    telemetry_.inside_preconditioner = was_inside;
    if (telemetry_.sync)
    {
      MFEM_DEVICE_SYNC;
    }
    const auto stop = Clock::now();
    ++telemetry_.preconditioner_apply_count;
    telemetry_.preconditioner_apply_seconds +=
      std::chrono::duration<double>(stop - start).count();
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return preconditioner_.GetMemoryClass();
  }

private:
  mfem::Solver& preconditioner_;
  KrylovApplyTelemetry& telemetry_;
};

struct ExactLocalFacePairCoupling
{
  int first_element = -1;
  int second_element = -1;
  int contributing_faces = 0;
  mfem::DenseMatrix first_to_second;
  mfem::DenseMatrix second_to_first;
};

class ExactL2DGFaceOperator final : public mfem::Operator
{
public:
  ExactL2DGFaceOperator(const mfem::ParFiniteElementSpace& sfes,
                        const mfem::FaceQuadratureSpace& fqs,
                        const mfem::Vector& side_q,
                        const mfem::IntegrationRule& shared_ir,
                        const mfem::Vector& shared_side_q,
                        mfem::real_t ip_sigma,
                        mfem::real_t kappa,
                        dg_face::DGDiffusionFaceScaling face_scaling)
    : mfem::Operator(sfes.GetTrueVSize()),
      sfes_(sfes),
      prolong_(sfes.GetProlongationMatrix()),
      restrict_(sfes.GetRestrictionOperator()),
      ip_sigma_(ip_sigma),
      kappa_(kappa),
      face_scaling_(face_scaling),
      sfes_sequence_(sfes.GetSequence()),
      mesh_sequence_(sfes.GetMesh() ? sfes.GetMesh()->GetSequence() : -1),
      local_x_gf_(const_cast<mfem::ParFiniteElementSpace*>(&sfes))
  {
    info_ = dg_face::BuildInteriorFaceDGDiffusionApplyData2D(
      sfes_,
      fqs,
      side_q,
      kappa,
      scratch_.shape,
      scratch_.weighted_normal_dshape,
      scratch_.penalty,
      scratch_.face_elements,
      scratch_.element_vdof_map,
      face_scaling,
      &scratch_.normal_dshape_base,
      &scratch_.penalty_metric);
    timing_enabled_ =
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_EXACT_FACE_TIMING",
                                      false);
    if (shared_side_q.Size() > 0)
    {
      shared_info_ = dg_face::BuildSharedFaceDGDiffusionApplyData2D(
        sfes_,
        shared_ir,
        shared_side_q,
        kappa,
        shared_scratch_.shape,
        shared_scratch_.weighted_normal_dshape,
        shared_scratch_.penalty,
        shared_scratch_.face_elements,
        shared_scratch_.element_vdof_map,
        shared_scratch_.face_nbr_dof_map,
        face_scaling,
        &shared_scratch_.normal_dshape_base,
        &shared_scratch_.penalty_metric);
      has_shared_ = (shared_info_.nf > 0);
    }
  }

  bool CanRefresh(const mfem::ParFiniteElementSpace& sfes,
                  const mfem::Vector& side_q,
                  const mfem::Vector& shared_side_q,
                  mfem::real_t kappa,
                  dg_face::DGDiffusionFaceScaling face_scaling) const
  {
    const mfem::Mesh* mesh = sfes.GetMesh();
    return &sfes == &sfes_ && sfes.GetSequence() == sfes_sequence_ && mesh &&
           mesh->GetSequence() == mesh_sequence_ && kappa == kappa_ &&
           face_scaling == face_scaling_ &&
           side_q.Size() == 2 * info_.nq * info_.nf &&
           shared_side_q.Size() == 2 * shared_info_.nq * shared_info_.nf &&
           (shared_info_.nf > 0) == has_shared_;
  }

  void RefreshCoefficients(const mfem::Vector& side_q,
                           const mfem::Vector& shared_side_q)
  {
    dg_face::LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
      info_, scratch_, side_q, kappa_, face_scaling_);
    if (has_shared_)
    {
      dg_face::LaunchSharedFaceDGDiffusionCoefficientUpdate2D(
        shared_info_,
        shared_scratch_,
        shared_side_q,
        kappa_,
        face_scaling_);
    }
  }

  bool TimingEnabled() const { return timing_enabled_; }

  void ResetTiming() const { timing_ = ExactFaceTimingStats{}; }

  const ExactFaceTimingStats& Timing() const { return timing_; }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return mfem::Device::GetDeviceMemoryClass();
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/exact-face/Mult");
    if (!timing_enabled_)
    {
      local_y_.SetSize(sfes_.GetVSize(), y);
      local_y_ = 0.0;

      if (prolong_)
      {
        ::dg_igr::NVTXRange prolong_range("IGR/exact-face/prolong");
        prolong_->Mult(x, local_x_gf_);
      }
      else
      {
        local_x_gf_ = x;
      }

      {
        ::dg_igr::NVTXRange apply_range("IGR/exact-face/local-apply");
        dg_face::LaunchInteriorFaceDGDiffusionApply2D(info_,
                                                      scratch_,
                                                      local_x_gf_,
                                                      ip_sigma_,
                                                      local_y_);
      }
      if (has_shared_)
      {
        {
          ::dg_igr::NVTXRange exchange_range("IGR/exact-face/shared-exchange");
          local_x_gf_.ExchangeFaceNbrData();
        }
        {
          ::dg_igr::NVTXRange apply_range("IGR/exact-face/shared-apply");
          dg_face::LaunchSharedFaceDGDiffusionApply2D(
            shared_info_,
            shared_scratch_,
            local_x_gf_,
            local_x_gf_.FaceNbrData(),
            ip_sigma_,
            local_y_);
        }
      }

      if (restrict_)
      {
        ::dg_igr::NVTXRange restrict_range("IGR/exact-face/restrict");
        restrict_->Mult(local_y_, y);
      }
      else
      {
        y = local_y_;
      }
      return;
    }

    const auto setup_start = Clock::now();
    local_y_.SetSize(sfes_.GetVSize(), y);
    local_y_ = 0.0;
    timing_.face_setup_seconds += ElapsedSeconds(setup_start);

    const auto prolong_start = Clock::now();
    if (prolong_)
    {
      ::dg_igr::NVTXRange prolong_range("IGR/exact-face/prolong");
      prolong_->Mult(x, local_x_gf_);
    }
    else
    {
      local_x_gf_ = x;
    }
    timing_.prolong_seconds += ElapsedSeconds(prolong_start);

    const auto local_apply_start = Clock::now();
    {
      ::dg_igr::NVTXRange apply_range("IGR/exact-face/local-apply");
      dg_face::LaunchInteriorFaceDGDiffusionApply2D(info_,
                                                    scratch_,
                                                    local_x_gf_,
                                                    ip_sigma_,
                                                    local_y_);
    }
    timing_.local_apply_seconds += ElapsedSeconds(local_apply_start);
    if (has_shared_)
    {
      const auto shared_exchange_start = Clock::now();
      {
        ::dg_igr::NVTXRange exchange_range("IGR/exact-face/shared-exchange");
        local_x_gf_.ExchangeFaceNbrData();
      }
      timing_.shared_exchange_seconds += ElapsedSeconds(shared_exchange_start);

      const auto shared_apply_start = Clock::now();
      {
        ::dg_igr::NVTXRange apply_range("IGR/exact-face/shared-apply");
        dg_face::LaunchSharedFaceDGDiffusionApply2D(shared_info_,
                                                    shared_scratch_,
                                                    local_x_gf_,
                                                    local_x_gf_.FaceNbrData(),
                                                    ip_sigma_,
                                                    local_y_);
      }
      timing_.shared_apply_seconds += ElapsedSeconds(shared_apply_start);
    }

    const auto restrict_start = Clock::now();
    if (restrict_)
    {
      ::dg_igr::NVTXRange restrict_range("IGR/exact-face/restrict");
      restrict_->Mult(local_y_, y);
    }
    else
    {
      y = local_y_;
    }
    timing_.restrict_seconds += ElapsedSeconds(restrict_start);
    ++timing_.mult_calls;
  }

  void AssembleDiagonal(mfem::Vector& diag) const override
  {
    MFEM_VERIFY(sfes_.GetTrueVSize() == sfes_.GetVSize(),
                "Exact L2 DG face diagonal currently requires true/local "
                "dof identity.");
    diag.SetSize(Height());
    diag.UseDevice(true);
    diag = 0.0;
    dg_face::AddInteriorFaceDGDiffusionDiagonal2D(info_,
                                                  scratch_,
                                                  ip_sigma_,
                                                  diag);
    if (has_shared_)
    {
      dg_face::AddSharedFaceDGDiffusionLocalDiagonal2D(shared_info_,
                                                       shared_scratch_,
                                                       ip_sigma_,
                                                       diag);
    }
  }

  void AddElementDiagonalBlocks(
    std::vector<mfem::DenseMatrix>& blocks) const
  {
    const int ne = sfes_.GetNE();
    const int dof = info_.dof;
    MFEM_VERIFY(static_cast<int>(blocks.size()) == ne && dof > 0,
                "Exact L2 DG face block assembly received inconsistent "
                "element storage.");
    for (const mfem::DenseMatrix& block : blocks)
    {
      MFEM_VERIFY(block.Height() == dof && block.Width() == dof,
                  "Exact L2 DG face block assembly received inconsistent "
                  "block dimensions.");
    }

    const auto add_side_block =
      [this, dof](int side,
                  int nq,
                  int face,
                  const mfem::Vector& shape,
                  const mfem::Vector& weighted_normal_dshape,
                  const mfem::Vector& penalty,
                  mfem::DenseMatrix& block)
    {
      const mfem::real_t* S = shape.HostRead();
      const mfem::real_t* N = weighted_normal_dshape.HostRead();
      const mfem::real_t* P = penalty.HostRead();
      for (int q = 0; q < nq; ++q)
      {
        for (int row = 0; row < dof; ++row)
        {
          const int row_index = side + 2 * (q + nq * (row + dof * face));
          const mfem::real_t sr = S[row_index];
          const mfem::real_t nr = N[row_index];
          for (int column = 0; column < dof; ++column)
          {
            const int column_index =
              side + 2 * (q + nq * (column + dof * face));
            const mfem::real_t sc = S[column_index];
            const mfem::real_t nc = N[column_index];
            if (side == 0)
            {
              block(row, column) +=
                -sr * nc + ip_sigma_ * nr * sc +
                P[q + nq * face] * sr * sc;
            }
            else
            {
              block(row, column) +=
                sr * nc - ip_sigma_ * nr * sc +
                P[q + nq * face] * sr * sc;
            }
          }
        }
      }
    };

    const auto local_faces =
      mfem::Reshape(scratch_.face_elements.HostRead(), 2, info_.nf);
    for (int face = 0; face < info_.nf; ++face)
    {
      for (int side = 0; side < 2; ++side)
      {
        const int element = local_faces(side, face);
        MFEM_VERIFY(element >= 0 && element < ne,
                    "Exact L2 DG face block assembly received an invalid "
                    "local element.");
        add_side_block(side,
                       info_.nq,
                       face,
                       scratch_.shape,
                       scratch_.weighted_normal_dshape,
                       scratch_.penalty,
                       blocks[static_cast<std::size_t>(element)]);
      }
    }

    if (has_shared_)
    {
      const int* shared_elements = shared_scratch_.face_elements.HostRead();
      for (int face = 0; face < shared_info_.nf; ++face)
      {
        const int element = shared_elements[face];
        MFEM_VERIFY(element >= 0 && element < ne,
                    "Exact L2 DG face block assembly received an invalid "
                    "shared-face element.");
        add_side_block(0,
                       shared_info_.nq,
                       face,
                       shared_scratch_.shape,
                       shared_scratch_.weighted_normal_dshape,
                       shared_scratch_.penalty,
                       blocks[static_cast<std::size_t>(element)]);
      }
    }
  }

  bool BuildLocalFacePairCouplings(
    std::vector<ExactLocalFacePairCoupling>& pairs,
    std::string& reason) const
  {
    pairs.clear();
    reason = "not-attempted";
    const int ne = sfes_.GetNE();
    const int dof = info_.dof;
    if (ne <= 1 || dof <= 0 || info_.nf <= 0)
    {
      reason = "requires-local-interior-face-pairs";
      return false;
    }

    const auto local_faces =
      mfem::Reshape(scratch_.face_elements.HostRead(), 2, info_.nf);
    const mfem::real_t* shape = scratch_.shape.HostRead();
    const mfem::real_t* normal =
      scratch_.weighted_normal_dshape.HostRead();
    const mfem::real_t* penalty = scratch_.penalty.HostRead();
    std::map<std::pair<int, int>, std::size_t> pair_index;

    for (int face = 0; face < info_.nf; ++face)
    {
      const int element0 = local_faces(0, face);
      const int element1 = local_faces(1, face);
      if (element0 < 0 || element0 >= ne || element1 < 0 || element1 >= ne)
      {
        reason = "invalid-local-face-element";
        return false;
      }
      if (element0 == element1)
      {
        reason = "self-coupled-local-face";
        return false;
      }

      const int first = std::min(element0, element1);
      const int second = std::max(element0, element1);
      const std::pair<int, int> key(first, second);
      auto found = pair_index.find(key);
      if (found == pair_index.end())
      {
        ExactLocalFacePairCoupling coupling;
        coupling.first_element = first;
        coupling.second_element = second;
        coupling.first_to_second.SetSize(dof);
        coupling.second_to_first.SetSize(dof);
        coupling.first_to_second = 0.0;
        coupling.second_to_first = 0.0;
        pairs.emplace_back(std::move(coupling));
        found = pair_index.emplace(key, pairs.size() - 1).first;
      }

      ExactLocalFacePairCoupling& coupling = pairs[found->second];
      ++coupling.contributing_faces;
      for (int q = 0; q < info_.nq; ++q)
      {
        const mfem::real_t p = penalty[q + info_.nq * face];
        for (int row = 0; row < dof; ++row)
        {
          const int row0 =
            0 + 2 * (q + info_.nq * (row + dof * face));
          const int row1 =
            1 + 2 * (q + info_.nq * (row + dof * face));
          const mfem::real_t s0r = shape[row0];
          const mfem::real_t s1r = shape[row1];
          const mfem::real_t n0r = normal[row0];
          const mfem::real_t n1r = normal[row1];
          for (int column = 0; column < dof; ++column)
          {
            const int column0 =
              0 + 2 * (q + info_.nq * (column + dof * face));
            const int column1 =
              1 + 2 * (q + info_.nq * (column + dof * face));
            const mfem::real_t s0c = shape[column0];
            const mfem::real_t s1c = shape[column1];
            const mfem::real_t n0c = normal[column0];
            const mfem::real_t n1c = normal[column1];

            // These are the exact off-diagonal blocks encoded by
            // InteriorFaceDGDiffusionApply2D: row side 0 / column side 1
            // and row side 1 / column side 0, respectively.
            const mfem::real_t block01 =
              -s0r * n1c - ip_sigma_ * n0r * s1c - p * s0r * s1c;
            const mfem::real_t block10 =
              s1r * n0c + ip_sigma_ * n1r * s0c - p * s1r * s0c;
            if (element0 == first)
            {
              coupling.first_to_second(row, column) += block01;
              coupling.second_to_first(row, column) += block10;
            }
            else
            {
              coupling.first_to_second(row, column) += block10;
              coupling.second_to_first(row, column) += block01;
            }
          }
        }
      }
    }

    if (pairs.empty())
    {
      reason = "empty-local-face-pair-set";
      return false;
    }
    reason = "active";
    return true;
  }

private:
  const mfem::ParFiniteElementSpace& sfes_;
  const mfem::Operator* prolong_ = nullptr;
  const mfem::Operator* restrict_ = nullptr;
  mfem::real_t ip_sigma_ = 0.0;
  mfem::real_t kappa_ = 0.0;
  dg_face::DGDiffusionFaceScaling face_scaling_ =
    dg_face::DGDiffusionFaceScaling::Arithmetic;
  long sfes_sequence_ = -1;
  long mesh_sequence_ = -1;
  dg_face::InteriorFaceDGDiffusionApplyInfo info_;
  dg_face::InteriorFaceDGDiffusionApplyScratch scratch_;
  bool has_shared_ = false;
  dg_face::SharedFaceDGDiffusionApplyInfo shared_info_;
  dg_face::SharedFaceDGDiffusionApplyScratch shared_scratch_;
  mutable mfem::Vector local_y_;
  mutable mfem::ParGridFunction local_x_gf_;
  bool timing_enabled_ = false;
  mutable ExactFaceTimingStats timing_;
};

class ExactFaceSumOperator final : public mfem::Operator
{
public:
  ExactFaceSumOperator(mfem::Operator* base,
                       mfem::Operator* face,
                       bool own_base = true,
                       bool own_face = true)
    : mfem::Operator(base->Height(), base->Width()),
      base_(base),
      face_(face),
      own_base_(own_base),
      own_face_(own_face),
      exact_face_(dynamic_cast<ExactL2DGFaceOperator*>(face)),
      timing_enabled_(exact_face_ != nullptr && exact_face_->TimingEnabled()),
      scratch_(base->Height())
  {
    MFEM_VERIFY(base_->Height() == face_->Height() &&
                  base_->Width() == face_->Width(),
                "Exact face sum operator received mismatched operators.");
  }

  ~ExactFaceSumOperator() override
  {
    if (own_base_)
    {
      delete base_;
    }
    if (own_face_)
    {
      delete face_;
    }
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return base_->GetMemoryClass() * face_->GetMemoryClass();
  }

  bool TimingEnabled() const { return timing_enabled_; }

  mfem::Operator& BaseOperator() const { return *base_; }
  mfem::Operator& FaceOperator() const { return *face_; }

  bool AddElementDiagonalFaceBlocks(
    std::vector<mfem::DenseMatrix>& blocks) const
  {
    if (!exact_face_)
    {
      return false;
    }
    exact_face_->AddElementDiagonalBlocks(blocks);
    return true;
  }

  bool BuildLocalFacePairCouplings(
    std::vector<ExactLocalFacePairCoupling>& pairs,
    std::string& reason) const
  {
    if (!exact_face_)
    {
      reason = "requires-exact-face-operator";
      return false;
    }
    return exact_face_->BuildLocalFacePairCouplings(pairs, reason);
  }

  void ResetTiming() const
  {
    operator_mult_calls_ = 0;
    timing_ = ExactFaceTimingStats{};
    if (exact_face_)
    {
      exact_face_->ResetTiming();
    }
  }

  long long LastMultCalls() const { return operator_mult_calls_; }

  void LogTimingSummary(MPI_Comm comm,
                        int solve_call,
                        int solve_iters,
                        double solve_seconds,
                        const std::string& precond_name) const
  {
    if (!timing_enabled_)
    {
      return;
    }

    ExactFaceTimingStats local = timing_;
    if (exact_face_)
    {
      const ExactFaceTimingStats& face_timing = exact_face_->Timing();
      local.face_setup_seconds = face_timing.face_setup_seconds;
      local.prolong_seconds = face_timing.prolong_seconds;
      local.local_apply_seconds = face_timing.local_apply_seconds;
      local.shared_exchange_seconds = face_timing.shared_exchange_seconds;
      local.shared_apply_seconds = face_timing.shared_apply_seconds;
      local.restrict_seconds = face_timing.restrict_seconds;
    }

    double seconds[] = {local.base_seconds,
                        local.face_seconds,
                        local.sum_axpy_seconds,
                        local.face_setup_seconds,
                        local.prolong_seconds,
                        local.local_apply_seconds,
                        local.shared_exchange_seconds,
                        local.shared_apply_seconds,
                        local.restrict_seconds};
    double max_seconds[sizeof(seconds) / sizeof(seconds[0])] = {};
    long long calls = local.mult_calls;
    long long max_calls = calls;
#ifdef MFEM_USE_MPI
    MPI_Allreduce(seconds,
                  max_seconds,
                  static_cast<int>(sizeof(seconds) / sizeof(seconds[0])),
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_MAX,
                  comm);
    MPI_Allreduce(&calls,
                  &max_calls,
                  1,
                  MPI_LONG_LONG,
                  MPI_MAX,
                  comm);
#else
    (void)comm;
    std::copy(seconds,
              seconds + static_cast<int>(sizeof(seconds) / sizeof(seconds[0])),
              max_seconds);
#endif
    if (!mfem::Mpi::Root())
    {
      return;
    }

    std::cout << "[igr-linear] event=exact-dg-face-timing"
              << " solve_call=" << solve_call
              << " operator_mult_calls=" << max_calls
              << " solve_iters=" << solve_iters
              << " solve_s=" << solve_seconds
              << " ranks=" << mfem::Mpi::WorldSize()
              << " precond=" << precond_name
              << " max_base_s=" << max_seconds[0]
              << " max_face_s=" << max_seconds[1]
              << " max_sum_axpy_s=" << max_seconds[2]
              << " max_face_setup_s=" << max_seconds[3]
              << " max_prolong_s=" << max_seconds[4]
              << " max_local_apply_s=" << max_seconds[5]
              << " max_shared_exchange_s=" << max_seconds[6]
              << " max_shared_apply_s=" << max_seconds[7]
              << " max_restrict_s=" << max_seconds[8]
              << " host_wall=1"
              << " sync_inserted=0"
              << std::endl;
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/exact-sum/Mult");
    ++operator_mult_calls_;
    if (!timing_enabled_)
    {
      {
        ::dg_igr::NVTXRange base_range("IGR/exact-sum/base");
        base_->Mult(x, y);
      }
      {
        ::dg_igr::NVTXRange face_range("IGR/exact-sum/face");
        face_->Mult(x, scratch_);
      }
      {
        ::dg_igr::NVTXRange axpy_range("IGR/exact-sum/axpy");
        y += scratch_;
      }
      return;
    }

    const auto base_start = Clock::now();
    {
      ::dg_igr::NVTXRange base_range("IGR/exact-sum/base");
      base_->Mult(x, y);
    }
    timing_.base_seconds += ElapsedSeconds(base_start);

    const auto face_start = Clock::now();
    {
      ::dg_igr::NVTXRange face_range("IGR/exact-sum/face");
      face_->Mult(x, scratch_);
    }
    timing_.face_seconds += ElapsedSeconds(face_start);

    const auto axpy_start = Clock::now();
    {
      ::dg_igr::NVTXRange axpy_range("IGR/exact-sum/axpy");
      y += scratch_;
    }
    timing_.sum_axpy_seconds += ElapsedSeconds(axpy_start);
    ++timing_.mult_calls;
  }

  void AssembleDiagonal(mfem::Vector& diag) const override
  {
    base_->AssembleDiagonal(diag);
    face_->AssembleDiagonal(scratch_);
    diag += scratch_;
  }

private:
  mfem::Operator* base_ = nullptr;
  mfem::Operator* face_ = nullptr;
  bool own_base_ = true;
  bool own_face_ = true;
  ExactL2DGFaceOperator* exact_face_ = nullptr;
  bool timing_enabled_ = false;
  mutable mfem::Vector scratch_;
  mutable long long operator_mult_calls_ = 0;
  mutable ExactFaceTimingStats timing_;
};

class ExactElementBlockJacobi final : public mfem::Solver
{
public:
  ExactElementBlockJacobi(
    int vector_size,
    int elements,
    int block_size,
    const std::vector<mfem::DenseMatrix>& inverse_blocks)
    : mfem::Solver(vector_size, false),
      elements_(elements),
      block_size_(block_size),
      inverse_blocks_(elements * block_size * block_size)
  {
    MFEM_VERIFY(vector_size > 0 && elements > 0 && block_size > 0 &&
                  vector_size == elements * block_size &&
                  static_cast<int>(inverse_blocks.size()) == elements,
                "Exact element-block Jacobi received invalid block storage.");
    double* blocks = inverse_blocks_.HostWrite();
    for (int element = 0; element < elements_; ++element)
    {
      const mfem::DenseMatrix& block =
        inverse_blocks[static_cast<std::size_t>(element)];
      MFEM_VERIFY(block.Height() == block_size_ &&
                    block.Width() == block_size_,
                  "Exact element-block Jacobi received inconsistent block "
                  "dimensions.");
      for (int column = 0; column < block_size_; ++column)
      {
        for (int row = 0; row < block_size_; ++row)
        {
          blocks[row + block_size_ * (column + block_size_ * element)] =
            block(row, column);
        }
      }
    }
    inverse_blocks_.UseDevice(true);
  }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                "Exact element-block Jacobi received a size-changing "
                "operator update.");
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    MFEM_VERIFY(x.Size() == Width(),
                "Exact element-block Jacobi received a mismatched input.");
    igr_device::LaunchExactElementBlockJacobiApply(
      elements_, block_size_, inverse_blocks_, x, y);
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    // Each stored inverse block is explicitly symmetrized at construction.
    Mult(x, y);
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return mfem::Device::GetDeviceMemoryClass();
  }

private:
  int elements_ = 0;
  int block_size_ = 0;
  mfem::Vector inverse_blocks_;
};

class ExactFacePairAdditiveSchwarz final : public mfem::Solver
{
public:
  ExactFacePairAdditiveSchwarz(
    int vector_size,
    int elements,
    int element_block_size,
    const std::vector<ExactLocalFacePairCoupling>& pairs,
    const std::vector<int>& overlap,
    const std::vector<mfem::DenseMatrix>& inverse_patches)
    : mfem::Solver(vector_size, false),
      elements_(elements),
      element_block_size_(element_block_size),
      patch_size_(2 * element_block_size),
      patches_(static_cast<int>(pairs.size())),
      patch_dofs_(patches_ * patch_size_),
      patch_weights_(patches_ * patch_size_),
      patch_inverse_(patches_ * patch_size_,
                     patches_,
                     patch_size_,
                     inverse_patches),
      patch_rhs_(patches_ * patch_size_),
      patch_solution_(patches_ * patch_size_)
  {
    MFEM_VERIFY(vector_size > 0 && elements_ > 1 &&
                  element_block_size_ > 0 &&
                  vector_size == elements_ * element_block_size_ &&
                  patches_ > 0 &&
                  static_cast<int>(overlap.size()) == elements_ &&
                  static_cast<int>(inverse_patches.size()) == patches_,
                "Exact face-pair Schwarz received inconsistent storage.");
    int* patch_dofs = patch_dofs_.HostWrite();
    mfem::real_t* patch_weights = patch_weights_.HostWrite();
    for (int patch = 0; patch < patches_; ++patch)
    {
      const ExactLocalFacePairCoupling& pair =
        pairs[static_cast<std::size_t>(patch)];
      MFEM_VERIFY(pair.first_element >= 0 &&
                    pair.first_element < elements_ &&
                    pair.second_element >= 0 &&
                    pair.second_element < elements_ &&
                    pair.first_element != pair.second_element,
                  "Exact face-pair Schwarz received an invalid patch.");
      const int patch_elements[2] = {pair.first_element,
                                     pair.second_element};
      for (int side = 0; side < 2; ++side)
      {
        const int element = patch_elements[side];
        MFEM_VERIFY(overlap[static_cast<std::size_t>(element)] > 0,
                    "Exact face-pair Schwarz requires positive overlap.");
        const mfem::real_t weight =
          1.0 / std::sqrt(static_cast<mfem::real_t>(
                  overlap[static_cast<std::size_t>(element)]));
        min_overlap_ =
          std::min(min_overlap_, overlap[static_cast<std::size_t>(element)]);
        max_overlap_ =
          std::max(max_overlap_, overlap[static_cast<std::size_t>(element)]);
        for (int local = 0; local < element_block_size_; ++local)
        {
          const int row =
            local + element_block_size_ * (side + 2 * patch);
          patch_dofs[row] = local + element_block_size_ * element;
          patch_weights[row] = weight;
        }
      }
    }
    patch_dofs_.GetMemory().UseDevice(true);
    patch_weights_.UseDevice(true);
    patch_rhs_.UseDevice(true);
    patch_solution_.UseDevice(true);
  }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                "Exact face-pair Schwarz received a size-changing operator "
                "update.");
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    MFEM_VERIFY(x.Size() == Width(),
                "Exact face-pair Schwarz received a mismatched input.");
    ::dg_igr::NVTXRange range("IGR/exact-face-pair-schwarz/Mult");
    const int patch_entries = patches_ * patch_size_;
    const int* patch_dofs = patch_dofs_.Read();
    const mfem::real_t* patch_weights = patch_weights_.Read();
    const mfem::real_t* input = x.Read();
    mfem::real_t* patch_rhs = patch_rhs_.Write();
    mfem::forall(patch_entries,
                 [=] MFEM_HOST_DEVICE(int entry)
                 {
                   patch_rhs[entry] =
                     patch_weights[entry] * input[patch_dofs[entry]];
                 });
    patch_inverse_.Mult(patch_rhs_, patch_solution_);
    y.SetSize(Width());
    y.UseDevice(true);
    y = 0.0;
    const mfem::real_t* patch_solution = patch_solution_.Read();
    mfem::real_t* output = y.ReadWrite();
    mfem::forall(patch_entries,
                 [=] MFEM_HOST_DEVICE(int entry)
                 {
                   ::AtomicAdd(output[patch_dofs[entry]],
                               patch_weights[entry] * patch_solution[entry]);
                 });
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    // P = R_w^T diag(A_patch^{-1}) R_w is symmetric because every patch
    // inverse is explicitly symmetrized after an FP64 Cholesky inversion.
    Mult(x, y);
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return mfem::Device::GetDeviceMemoryClass();
  }

  int PatchCount() const { return patches_; }
  int PatchSize() const { return patch_size_; }
  int MinOverlap() const { return min_overlap_; }
  int MaxOverlap() const { return max_overlap_; }
  long long RestrictionNonzeros() const
  {
    return static_cast<long long>(patches_) * patch_size_;
  }
  long long InverseEntries() const
  {
    return static_cast<long long>(patches_) * patch_size_ * patch_size_;
  }

private:
  int elements_ = 0;
  int element_block_size_ = 0;
  int patch_size_ = 0;
  int patches_ = 0;
  int min_overlap_ = std::numeric_limits<int>::max();
  int max_overlap_ = 0;
  mfem::Array<int> patch_dofs_;
  mfem::Vector patch_weights_;
  ExactElementBlockJacobi patch_inverse_;
  mutable mfem::Vector patch_rhs_;
  mutable mfem::Vector patch_solution_;
};

class SharedCorrelationBlockJacobi final : public mfem::Solver
{
public:
  SharedCorrelationBlockJacobi(
    int vector_size,
    int elements,
    int block_size,
    const mfem::DenseMatrix& inverse_correlation,
    const std::vector<mfem::real_t>& inverse_sqrt_diagonal)
    : mfem::Solver(vector_size, false),
      elements_(elements),
      block_size_(block_size),
      inverse_correlation_(block_size, block_size, 1),
      inverse_sqrt_diagonal_(vector_size),
      scaled_(vector_size)
  {
    MFEM_VERIFY(vector_size > 0 && elements > 0 && block_size > 0 &&
                  vector_size == elements * block_size &&
                  inverse_correlation.Height() == block_size &&
                  inverse_correlation.Width() == block_size &&
                  static_cast<int>(inverse_sqrt_diagonal.size()) ==
                    vector_size,
                "Shared-correlation block Jacobi received inconsistent "
                "storage.");
    mfem::real_t* matrix = inverse_correlation_.HostWrite();
    for (int column = 0; column < block_size_; ++column)
    {
      for (int row = 0; row < block_size_; ++row)
      {
        matrix[row + block_size_ * column] =
          inverse_correlation(row, column);
      }
    }
    std::copy(inverse_sqrt_diagonal.begin(),
              inverse_sqrt_diagonal.end(),
              inverse_sqrt_diagonal_.HostWrite());
    inverse_correlation_.Read();
    inverse_sqrt_diagonal_.UseDevice(true);
    scaled_.UseDevice(true);
  }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                "Shared-correlation block Jacobi received a size-changing "
                "operator update.");
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    MFEM_VERIFY(x.Size() == Width(),
                "Shared-correlation block Jacobi received a mismatched "
                "input.");
    scaled_ = x;
    scaled_ *= inverse_sqrt_diagonal_;
    y.SetSize(Height());
    y.UseDevice(true);
    // With one shared matrix, the GPU-BLAS backend sees X as an
    // (block_size x elements) panel and issues one true GEMM rather than a
    // batch of unrelated one-RHS GEMVs.
    mfem::BatchedLinAlg::Mult(inverse_correlation_, scaled_, y);
    y *= inverse_sqrt_diagonal_;
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    Mult(x, y);
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return mfem::Device::GetDeviceMemoryClass();
  }

private:
  int elements_ = 0;
  int block_size_ = 0;
  mfem::DenseTensor inverse_correlation_;
  mfem::Vector inverse_sqrt_diagonal_;
  mutable mfem::Vector scaled_;
};

// Symmetric diagonal congruence for a persistent approximate inverse.  If the
// current stage operator is approximately A_k = D A_0 D, this applies
// D^{-1} A_0^{-1} D^{-1}.  The exact stage operator and stopping certificate
// are unchanged; this is an env-only screen for coefficient-aware reuse of a
// static LOR hierarchy.
class SymmetricDiagonalScaledSolver final : public mfem::Solver
{
public:
  SymmetricDiagonalScaledSolver(mfem::Solver& base,
                                const mfem::Vector& base_diagonal,
                                const mfem::Operator& current_operator)
    : mfem::Solver(base.Height(), false),
      base_(&base),
      scale_(base.Height()),
      scratch_(base.Height())
  {
    MFEM_VERIFY(base.Height() == base.Width() &&
                  base_diagonal.Size() == base.Height() &&
                  current_operator.Height() == base.Height() &&
                  current_operator.Width() == base.Width(),
                "Scaled LOR preconditioner received inconsistent sizes.");
    mfem::Vector current_diagonal(base.Height());
    current_operator.AssembleDiagonal(current_diagonal);
    const mfem::real_t* base_data = base_diagonal.HostRead();
    const mfem::real_t* current_data = current_diagonal.HostRead();
    mfem::real_t* scale_data = scale_.HostWrite();
    for (int i = 0; i < scale_.Size(); ++i)
    {
      MFEM_VERIFY(std::isfinite(base_data[i]) && base_data[i] > 0.0 &&
                    std::isfinite(current_data[i]) && current_data[i] > 0.0,
                  "Scaled LOR preconditioner requires positive diagonals.");
      scale_data[i] = std::sqrt(base_data[i] / current_data[i]);
    }
    scale_.UseDevice(true);
    scratch_.UseDevice(true);
  }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                "Scaled LOR preconditioner received a size-changing update.");
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    scratch_ = x;
    scratch_ *= scale_;
    base_->Mult(scratch_, y);
    y *= scale_;
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    Mult(x, y);
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return mfem::Device::GetDeviceMemoryClass();
  }

private:
  mfem::Solver* base_ = nullptr;
  mfem::Vector scale_;
  mutable mfem::Vector scratch_;
};

class AdditiveOwnedSolver final : public mfem::Solver
{
public:
  AdditiveOwnedSolver(std::unique_ptr<mfem::Solver> first,
                      std::unique_ptr<mfem::Solver> second)
    : mfem::Solver(first ? first->Height() : 0, false),
      first_(std::move(first)),
      second_(std::move(second)),
      scratch_(Height())
  {
    MFEM_VERIFY(first_ && second_ && Height() == Width() &&
                  second_->Height() == Height() &&
                  second_->Width() == Width(),
                "Additive preconditioner received inconsistent solvers.");
    scratch_.UseDevice(true);
  }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                "Additive preconditioner received a size-changing update.");
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    first_->Mult(x, y);
    second_->Mult(x, scratch_);
    y += scratch_;
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    Mult(x, y);
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return mfem::Device::GetDeviceMemoryClass();
  }

private:
  std::unique_ptr<mfem::Solver> first_;
  std::unique_ptr<mfem::Solver> second_;
  mutable mfem::Vector scratch_;
};

class ElementMeanZeroSolver final : public mfem::Solver
{
public:
  ElementMeanZeroSolver(std::unique_ptr<mfem::Solver> base,
                        int elements,
                        int block_size)
    : mfem::Solver(base ? base->Height() : 0, false),
      base_(std::move(base)),
      elements_(elements),
      block_size_(block_size),
      projected_(Height()),
      applied_(Height())
  {
    MFEM_VERIFY(base_ && elements_ > 0 && block_size_ > 1 &&
                  Height() == elements_ * block_size_,
                "Mean-zero element solver received inconsistent storage.");
    projected_.UseDevice(true);
    applied_.UseDevice(true);
  }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                "Mean-zero element solver received a size-changing update.");
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    igr_device::LaunchElementMeanProjection(
      elements_, block_size_, x, projected_);
    base_->Mult(projected_, applied_);
    igr_device::LaunchElementMeanProjection(
      elements_, block_size_, applied_, y);
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    Mult(x, y);
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return mfem::Device::GetDeviceMemoryClass();
  }

private:
  std::unique_ptr<mfem::Solver> base_;
  int elements_ = 0;
  int block_size_ = 0;
  mutable mfem::Vector projected_;
  mutable mfem::Vector applied_;
};

class OriginalElementDG0Correction final : public mfem::Solver
{
public:
  OriginalElementDG0Correction(
    int high_order_size,
    int elements,
    int block_size,
    std::unique_ptr<mfem::DG_FECollection> coarse_collection,
    std::unique_ptr<mfem::ParFiniteElementSpace> coarse_space,
    std::unique_ptr<mfem::ParBilinearForm> coarse_form,
    std::unique_ptr<mfem::HypreParMatrix> coarse_matrix,
    std::unique_ptr<mfem::HypreBoomerAMG> coarse_solver)
    : mfem::Solver(high_order_size, false),
      elements_(elements),
      block_size_(block_size),
      coarse_collection_(std::move(coarse_collection)),
      coarse_space_(std::move(coarse_space)),
      coarse_form_(std::move(coarse_form)),
      coarse_matrix_(std::move(coarse_matrix)),
      coarse_solver_(std::move(coarse_solver)),
      coarse_rhs_(elements),
      coarse_solution_(elements)
  {
    MFEM_VERIFY(elements_ > 0 && block_size_ > 1 &&
                  Height() == elements_ * block_size_ &&
                  coarse_collection_ && coarse_space_ && coarse_form_ &&
                  coarse_matrix_ && coarse_solver_ &&
                  coarse_space_->GetTrueVSize() == elements_ &&
                  coarse_matrix_->Height() == elements_ &&
                  coarse_matrix_->Width() == elements_,
                "Original-element DG0 correction received inconsistent "
                "storage.");
    coarse_rhs_.UseDevice(true);
    coarse_solution_.UseDevice(true);
  }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                "Original-element DG0 correction received a size-changing "
                "update.");
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    igr_device::LaunchElementConstantRestriction(
      elements_, block_size_, x, coarse_rhs_);
    coarse_solver_->Mult(coarse_rhs_, coarse_solution_);
    igr_device::LaunchElementConstantProlongation(
      elements_, block_size_, coarse_solution_, y);
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    Mult(x, y);
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return mfem::Device::GetDeviceMemoryClass();
  }

private:
  int elements_ = 0;
  int block_size_ = 0;
  std::unique_ptr<mfem::DG_FECollection> coarse_collection_;
  std::unique_ptr<mfem::ParFiniteElementSpace> coarse_space_;
  std::unique_ptr<mfem::ParBilinearForm> coarse_form_;
  std::unique_ptr<mfem::HypreParMatrix> coarse_matrix_;
  std::unique_ptr<mfem::HypreBoomerAMG> coarse_solver_;
  mutable mfem::Vector coarse_rhs_;
  mutable mfem::Vector coarse_solution_;
};

struct ExactElementBlockJacobiBuild
{
  std::unique_ptr<mfem::Solver> solver;
  std::string reason = "not-attempted";
  int colors = 0;
  int block_size = 0;
  int patch_count = 0;
  int patch_size = 0;
  int min_overlap = 0;
  int max_overlap = 0;
  int max_faces_per_pair = 0;
  long long operator_applies = 0;
  long long restriction_nonzeros = 0;
  long long inverse_entries = 0;
  double max_symmetry_defect = 0.0;
  double max_inverse_residual = 0.0;
};

bool InvertSymmetricPositiveBlock(const mfem::DenseMatrix& input,
                                  mfem::DenseMatrix& inverse,
                                  double& relative_inverse_residual,
                                  std::string& reason)
{
  const int n = input.Height();
  if (n <= 0 || input.Width() != n)
  {
    reason = "invalid-block-size";
    return false;
  }

  std::vector<double> diagonal_scale(static_cast<std::size_t>(n), 0.0);
  double max_diagonal = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const double diagonal = input(i, i);
    if (!std::isfinite(diagonal) || diagonal <= 0.0)
    {
      reason = "nonpositive-block-diagonal";
      return false;
    }
    diagonal_scale[static_cast<std::size_t>(i)] = std::sqrt(diagonal);
    max_diagonal = std::max(max_diagonal, diagonal);
  }
  if (!std::isfinite(max_diagonal) || max_diagonal <= 0.0)
  {
    reason = "invalid-block-scale";
    return false;
  }

  // Cholesky-factor the symmetrically diagonally scaled block. Scaling its
  // diagonal to one avoids treating small physical units as small pivots.
  std::vector<double> lower(static_cast<std::size_t>(n * n), 0.0);
  const double pivot_floor =
    256.0 * std::numeric_limits<double>::epsilon() *
    static_cast<double>(n);
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      double value = input(i, j) /
                     (diagonal_scale[static_cast<std::size_t>(i)] *
                      diagonal_scale[static_cast<std::size_t>(j)]);
      for (int k = 0; k < j; ++k)
      {
        value -= lower[static_cast<std::size_t>(i * n + k)] *
                 lower[static_cast<std::size_t>(j * n + k)];
      }
      if (i == j)
      {
        if (!std::isfinite(value) || value <= pivot_floor)
        {
          reason = "block-cholesky-failed";
          return false;
        }
        lower[static_cast<std::size_t>(i * n + j)] = std::sqrt(value);
      }
      else
      {
        const double pivot =
          lower[static_cast<std::size_t>(j * n + j)];
        value /= pivot;
        if (!std::isfinite(value))
        {
          reason = "nonfinite-block-factor";
          return false;
        }
        lower[static_cast<std::size_t>(i * n + j)] = value;
      }
    }
  }

  mfem::DenseMatrix scaled_inverse(n);
  scaled_inverse = 0.0;
  std::vector<double> forward(static_cast<std::size_t>(n), 0.0);
  std::vector<double> solution(static_cast<std::size_t>(n), 0.0);
  for (int column = 0; column < n; ++column)
  {
    std::fill(forward.begin(), forward.end(), 0.0);
    std::fill(solution.begin(), solution.end(), 0.0);
    for (int i = 0; i < n; ++i)
    {
      double value = (i == column) ? 1.0 : 0.0;
      for (int k = 0; k < i; ++k)
      {
        value -= lower[static_cast<std::size_t>(i * n + k)] *
                 forward[static_cast<std::size_t>(k)];
      }
      value /= lower[static_cast<std::size_t>(i * n + i)];
      forward[static_cast<std::size_t>(i)] = value;
    }
    for (int i = n - 1; i >= 0; --i)
    {
      double value = forward[static_cast<std::size_t>(i)];
      for (int k = i + 1; k < n; ++k)
      {
        value -= lower[static_cast<std::size_t>(k * n + i)] *
                 solution[static_cast<std::size_t>(k)];
      }
      value /= lower[static_cast<std::size_t>(i * n + i)];
      solution[static_cast<std::size_t>(i)] = value;
    }
    for (int i = 0; i < n; ++i)
    {
      scaled_inverse(i, column) = solution[static_cast<std::size_t>(i)];
    }
  }

  inverse.SetSize(n);
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      inverse(i, j) =
        scaled_inverse(i, j) /
        (diagonal_scale[static_cast<std::size_t>(i)] *
         diagonal_scale[static_cast<std::size_t>(j)]);
      if (!std::isfinite(inverse(i, j)))
      {
        reason = "nonfinite-block-inverse";
        return false;
      }
    }
  }
  // Preserve exact symmetry in the stored preconditioner even when the two
  // triangular solves differ by the final roundoff bit.
  for (int i = 0; i < n; ++i)
  {
    for (int j = i + 1; j < n; ++j)
    {
      const double symmetric = 0.5 * (inverse(i, j) + inverse(j, i));
      inverse(i, j) = symmetric;
      inverse(j, i) = symmetric;
    }
  }

  double norm_input = 0.0;
  double norm_inverse = 0.0;
  double norm_residual = 0.0;
  for (int i = 0; i < n; ++i)
  {
    double input_row_sum = 0.0;
    double inverse_row_sum = 0.0;
    double residual_row_sum = 0.0;
    for (int j = 0; j < n; ++j)
    {
      input_row_sum += std::abs(input(i, j));
      inverse_row_sum += std::abs(inverse(i, j));
      double product = 0.0;
      for (int k = 0; k < n; ++k)
      {
        product += input(i, k) * inverse(k, j);
      }
      residual_row_sum +=
        std::abs(product - ((i == j) ? 1.0 : 0.0));
    }
    norm_input = std::max(norm_input, input_row_sum);
    norm_inverse = std::max(norm_inverse, inverse_row_sum);
    norm_residual = std::max(norm_residual, residual_row_sum);
  }
  const double residual_scale = norm_input * norm_inverse;
  if (!std::isfinite(norm_input) || !std::isfinite(norm_inverse) ||
      !std::isfinite(norm_residual) || !std::isfinite(residual_scale))
  {
    reason = "nonfinite-block-inverse-validation";
    return false;
  }
  relative_inverse_residual =
    norm_residual / std::max(1.0, residual_scale);
  if (!std::isfinite(relative_inverse_residual) ||
      relative_inverse_residual > 1.0e-10)
  {
    reason = "block-inverse-residual";
    return false;
  }
  for (int i = 0; i < n; ++i)
  {
    if (!std::isfinite(inverse(i, i)) || inverse(i, i) <= 0.0)
    {
      reason = "nonpositive-inverse-diagonal";
      return false;
    }
  }
  return true;
}

ExactElementBlockJacobiBuild BuildExactElementBlockJacobi(
  const mfem::ParFiniteElementSpace& sfes,
  ExactFaceSumOperator& op,
  bool shared_correlation,
  bool face_pair_schwarz)
{
  ExactElementBlockJacobiBuild result;
  mfem::Mesh* mesh = sfes.GetMesh();
  if (!mesh || sfes.GetNE() <= 0)
  {
    result.reason = "invalid-mesh";
    return result;
  }
  if (mesh->Nonconforming())
  {
    result.reason = "requires-conforming-mesh";
    return result;
  }

  const int ne = sfes.GetNE();
  const int vector_size = sfes.GetVSize();
  if (op.Height() != vector_size || op.Width() != vector_size)
  {
    result.reason = "operator-size-mismatch";
    return result;
  }

  std::vector<std::vector<int>> element_dofs(static_cast<std::size_t>(ne));
  std::vector<int> dof_owner(static_cast<std::size_t>(vector_size), -1);
  mfem::Array<int> vdofs;
  int block_size = -1;
  for (int element = 0; element < ne; ++element)
  {
    sfes.GetElementVDofs(element, vdofs);
    if (block_size < 0)
    {
      block_size = vdofs.Size();
    }
    if (block_size <= 0 || vdofs.Size() != block_size)
    {
      result.reason = "nonuniform-element-block-size";
      return result;
    }
    auto& dofs = element_dofs[static_cast<std::size_t>(element)];
    dofs.resize(static_cast<std::size_t>(block_size));
    for (int local = 0; local < block_size; ++local)
    {
      const int dof = vdofs[local];
      if (dof < 0 || dof >= vector_size)
      {
        result.reason = "signed-or-invalid-element-dof";
        return result;
      }
      if (dof_owner[static_cast<std::size_t>(dof)] >= 0)
      {
        result.reason = "non-discontinuous-element-dofs";
        return result;
      }
      if (dof != local + block_size * element)
      {
        result.reason = "requires-contiguous-element-dofs";
        return result;
      }
      dof_owner[static_cast<std::size_t>(dof)] = element;
      dofs[static_cast<std::size_t>(local)] = dof;
    }
  }
  if (std::find(dof_owner.begin(), dof_owner.end(), -1) != dof_owner.end())
  {
    result.reason = "element-dofs-do-not-cover-vector";
    return result;
  }

  // The base operator is element-local. Seed the same local basis column in
  // every element and recover all base principal blocks at once. The exact
  // face operator contributes its self blocks directly from persistent face
  // quadrature data, avoiding graph-colored complete-operator applications.
  result.colors = 1;
  result.block_size = block_size;

  std::vector<mfem::DenseMatrix> blocks;
  blocks.reserve(static_cast<std::size_t>(ne));
  for (int element = 0; element < ne; ++element)
  {
    blocks.emplace_back(block_size);
    blocks.back() = 0.0;
  }

  mfem::Vector probe(vector_size);
  mfem::Vector response(vector_size);
  for (int column = 0; column < block_size; ++column)
  {
    double* probe_data = probe.HostWrite();
    std::fill(probe_data, probe_data + vector_size, 0.0);
    for (int element = 0; element < ne; ++element)
    {
      const int dof = element_dofs[static_cast<std::size_t>(element)]
                                  [static_cast<std::size_t>(column)];
      probe_data[dof] = 1.0;
    }
    op.BaseOperator().Mult(probe, response);
    ++result.operator_applies;
    const double* response_data = response.HostRead();
    for (int element = 0; element < ne; ++element)
    {
      mfem::DenseMatrix& block = blocks[static_cast<std::size_t>(element)];
      const auto& dofs = element_dofs[static_cast<std::size_t>(element)];
      for (int row = 0; row < block_size; ++row)
      {
        block(row, column) =
          response_data[dofs[static_cast<std::size_t>(row)]];
      }
    }
  }
  if (!op.AddElementDiagonalFaceBlocks(blocks))
  {
    result.reason = "requires-exact-face-block-data";
    return result;
  }

  for (int element = 0; element < ne; ++element)
  {
    mfem::DenseMatrix& block = blocks[static_cast<std::size_t>(element)];
    double max_entry = 0.0;
    double max_asymmetry = 0.0;
    for (int i = 0; i < block_size; ++i)
    {
      for (int j = 0; j < block_size; ++j)
      {
        if (!std::isfinite(block(i, j)))
        {
          result.reason = "nonfinite-probed-block";
          return result;
        }
        max_entry = std::max(max_entry, std::abs(block(i, j)));
        max_asymmetry =
          std::max(max_asymmetry, std::abs(block(i, j) - block(j, i)));
      }
    }
    const double symmetry_defect =
      max_asymmetry / std::max(max_entry, 1.0e-300);
    result.max_symmetry_defect =
      std::max(result.max_symmetry_defect, symmetry_defect);
    if (!std::isfinite(symmetry_defect) || symmetry_defect > 1.0e-9)
    {
      result.reason = "probed-block-not-symmetric";
      return result;
    }
    for (int i = 0; i < block_size; ++i)
    {
      for (int j = i + 1; j < block_size; ++j)
      {
        const double symmetric = 0.5 * (block(i, j) + block(j, i));
        block(i, j) = symmetric;
        block(j, i) = symmetric;
      }
    }

  }

  if (face_pair_schwarz)
  {
    if (shared_correlation)
    {
      result.reason = "face-pair-excludes-shared-correlation";
      return result;
    }
    if (std::numeric_limits<mfem::real_t>::digits <
        std::numeric_limits<double>::digits)
    {
      result.reason = "face-pair-requires-fp64-real";
      return result;
    }
    int comm_size = 1;
#ifdef MFEM_USE_MPI
    MPI_Comm_size(sfes.GetComm(), &comm_size);
#endif
    if (comm_size != 1)
    {
      // The minimum oracle deliberately avoids pretending that a local-only
      // pair cover is the same method under a different MPI partition. A
      // production distributed Schwarz version needs explicit shared-patch
      // ownership, halo restriction, and the same symmetric weights on both
      // ranks.
      result.reason = "face-pair-screen-requires-one-rank";
      return result;
    }

    std::vector<ExactLocalFacePairCoupling> pairs;
    std::string pair_reason;
    if (!op.BuildLocalFacePairCouplings(pairs, pair_reason))
    {
      result.reason = "face-pair-" + pair_reason;
      return result;
    }

    std::vector<int> overlap(static_cast<std::size_t>(ne), 0);
    result.patch_count = static_cast<int>(pairs.size());
    result.patch_size = 2 * block_size;
    for (const ExactLocalFacePairCoupling& pair : pairs)
    {
      if (pair.first_element < 0 || pair.first_element >= ne ||
          pair.second_element < 0 || pair.second_element >= ne ||
          pair.first_element == pair.second_element ||
          pair.first_to_second.Height() != block_size ||
          pair.first_to_second.Width() != block_size ||
          pair.second_to_first.Height() != block_size ||
          pair.second_to_first.Width() != block_size)
      {
        result.reason = "invalid-face-pair-coupling";
        return result;
      }
      ++overlap[static_cast<std::size_t>(pair.first_element)];
      ++overlap[static_cast<std::size_t>(pair.second_element)];
      result.max_faces_per_pair =
        std::max(result.max_faces_per_pair, pair.contributing_faces);
    }
    result.min_overlap = std::numeric_limits<int>::max();
    for (int count : overlap)
    {
      if (count <= 0)
      {
        result.reason = "face-pair-cover-misses-element";
        return result;
      }
      result.min_overlap = std::min(result.min_overlap, count);
      result.max_overlap = std::max(result.max_overlap, count);
    }

    std::vector<mfem::DenseMatrix> inverse_patches;
    inverse_patches.reserve(pairs.size());
    for (const ExactLocalFacePairCoupling& pair : pairs)
    {
      mfem::DenseMatrix patch(result.patch_size);
      patch = 0.0;
      const mfem::DenseMatrix& first_diagonal =
        blocks[static_cast<std::size_t>(pair.first_element)];
      const mfem::DenseMatrix& second_diagonal =
        blocks[static_cast<std::size_t>(pair.second_element)];
      for (int column = 0; column < block_size; ++column)
      {
        for (int row = 0; row < block_size; ++row)
        {
          patch(row, column) = first_diagonal(row, column);
          patch(row + block_size, column + block_size) =
            second_diagonal(row, column);
          patch(row, column + block_size) =
            pair.first_to_second(row, column);
          patch(row + block_size, column) =
            pair.second_to_first(row, column);
        }
      }

      double max_entry = 0.0;
      double max_asymmetry = 0.0;
      for (int i = 0; i < result.patch_size; ++i)
      {
        for (int j = 0; j < result.patch_size; ++j)
        {
          if (!std::isfinite(patch(i, j)))
          {
            result.reason = "nonfinite-face-pair-patch";
            return result;
          }
          max_entry = std::max(max_entry, std::abs(patch(i, j)));
          max_asymmetry =
            std::max(max_asymmetry,
                     std::abs(patch(i, j) - patch(j, i)));
        }
      }
      const double symmetry_defect =
        max_asymmetry / std::max(max_entry, 1.0e-300);
      result.max_symmetry_defect =
        std::max(result.max_symmetry_defect, symmetry_defect);
      if (!std::isfinite(symmetry_defect) || symmetry_defect > 1.0e-9)
      {
        result.reason = "face-pair-patch-not-symmetric";
        return result;
      }
      for (int i = 0; i < result.patch_size; ++i)
      {
        for (int j = i + 1; j < result.patch_size; ++j)
        {
          const double symmetric = 0.5 * (patch(i, j) + patch(j, i));
          patch(i, j) = symmetric;
          patch(j, i) = symmetric;
        }
      }

      mfem::DenseMatrix inverse;
      double inverse_residual = 0.0;
      std::string inversion_reason;
      if (!InvertSymmetricPositiveBlock(patch,
                                        inverse,
                                        inverse_residual,
                                        inversion_reason))
      {
        result.reason = "face-pair-" + inversion_reason;
        return result;
      }
      result.max_inverse_residual =
        std::max(result.max_inverse_residual, inverse_residual);
      inverse_patches.emplace_back(std::move(inverse));
    }

    result.restriction_nonzeros =
      static_cast<long long>(result.patch_count) * result.patch_size;
    result.inverse_entries =
      static_cast<long long>(result.patch_count) * result.patch_size *
      result.patch_size;
    result.solver = std::make_unique<ExactFacePairAdditiveSchwarz>(
      vector_size,
      ne,
      block_size,
      pairs,
      overlap,
      inverse_patches);
    result.reason = "active-face-pair";
    return result;
  }

  if (shared_correlation)
  {
    std::vector<mfem::real_t> inverse_sqrt_diagonal(
      static_cast<std::size_t>(vector_size), 0.0);
    std::vector<double> local_correlation_sum(
      static_cast<std::size_t>(block_size * block_size), 0.0);
    for (int element = 0; element < ne; ++element)
    {
      const mfem::DenseMatrix& block =
        blocks[static_cast<std::size_t>(element)];
      for (int i = 0; i < block_size; ++i)
      {
        const double diagonal = block(i, i);
        if (!std::isfinite(diagonal) || diagonal <= 0.0)
        {
          result.reason = "nonpositive-correlation-block-diagonal";
          return result;
        }
        inverse_sqrt_diagonal[static_cast<std::size_t>(
          i + block_size * element)] = 1.0 / std::sqrt(diagonal);
      }
      for (int column = 0; column < block_size; ++column)
      {
        const double column_scale = inverse_sqrt_diagonal[static_cast<
          std::size_t>(column + block_size * element)];
        for (int row = 0; row < block_size; ++row)
        {
          const double row_scale = inverse_sqrt_diagonal[static_cast<
            std::size_t>(row + block_size * element)];
          local_correlation_sum[static_cast<std::size_t>(
            row + block_size * column)] +=
            row_scale * block(row, column) * column_scale;
        }
      }
    }

    std::vector<double> global_correlation_sum(
      local_correlation_sum.size(), 0.0);
    long long local_elements = ne;
    long long global_elements = 0;
#ifdef MFEM_USE_MPI
    MPI_Allreduce(local_correlation_sum.data(),
                  global_correlation_sum.data(),
                  block_size * block_size,
                  MPI_DOUBLE,
                  MPI_SUM,
                  sfes.GetComm());
    MPI_Allreduce(&local_elements,
                  &global_elements,
                  1,
                  MPI_LONG_LONG,
                  MPI_SUM,
                  sfes.GetComm());
#else
    global_correlation_sum = local_correlation_sum;
    global_elements = local_elements;
#endif
    if (global_elements <= 0)
    {
      result.reason = "empty-global-correlation-block";
      return result;
    }

    mfem::DenseMatrix correlation(block_size);
    for (int column = 0; column < block_size; ++column)
    {
      for (int row = 0; row < block_size; ++row)
      {
        correlation(row, column) =
          global_correlation_sum[static_cast<std::size_t>(
            row + block_size * column)] /
          static_cast<double>(global_elements);
      }
    }
    for (int i = 0; i < block_size; ++i)
    {
      for (int j = i + 1; j < block_size; ++j)
      {
        const double symmetric =
          0.5 * (correlation(i, j) + correlation(j, i));
        correlation(i, j) = symmetric;
        correlation(j, i) = symmetric;
      }
    }

    mfem::DenseMatrix inverse_correlation;
    double inverse_residual = 0.0;
    std::string inversion_reason;
    if (!InvertSymmetricPositiveBlock(correlation,
                                      inverse_correlation,
                                      inverse_residual,
                                      inversion_reason))
    {
      result.reason = "correlation-" + inversion_reason;
      return result;
    }
    result.max_inverse_residual = inverse_residual;
    result.solver = std::make_unique<SharedCorrelationBlockJacobi>(
      vector_size,
      ne,
      block_size,
      inverse_correlation,
      inverse_sqrt_diagonal);
    result.reason = "active-shared-correlation";
    return result;
  }

  std::vector<mfem::DenseMatrix> inverse_blocks;
  inverse_blocks.reserve(static_cast<std::size_t>(ne));
  for (int element = 0; element < ne; ++element)
  {
    mfem::DenseMatrix block_inverse;
    double inverse_residual = 0.0;
    std::string inversion_reason;
    if (!InvertSymmetricPositiveBlock(blocks[static_cast<std::size_t>(element)],
                                      block_inverse,
                                      inverse_residual,
                                      inversion_reason))
    {
      result.reason = inversion_reason;
      return result;
    }
    result.max_inverse_residual =
      std::max(result.max_inverse_residual, inverse_residual);
    inverse_blocks.emplace_back(block_inverse);
  }
  result.solver = std::make_unique<ExactElementBlockJacobi>(vector_size,
                                                            ne,
                                                            block_size,
                                                            inverse_blocks);
  result.reason = "active";
  return result;
}

} // namespace

const mfem::Operator* IGRSolver::GetLinearBaseOperator() const
{
  const auto* exact_sum =
    dynamic_cast<const ExactFaceSumOperator*>(ActiveLinearOperator_());
  return exact_sum ? &exact_sum->BaseOperator() : nullptr;
}

const mfem::Operator* IGRSolver::GetLinearFaceOperator() const
{
  const auto* exact_sum =
    dynamic_cast<const ExactFaceSumOperator*>(ActiveLinearOperator_());
  return exact_sum ? &exact_sum->FaceOperator() : nullptr;
}

void IGRSolver::BuildBilinearForm_()
{
  // Any existing exact-face sum refers to the old bilinear form and face
  // layout. Drop both before rebuilding the coefficient-dependent forms.
  Ah_.Clear();
  exact_face_operator_.reset();

  const bool debug_qf =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=build-bilinear-start" << "\n";
  }
  a_ = std::make_unique<mfem::ParBilinearForm>(&sfes_);
  use_partial_assembly_ = ShouldUsePartialAssembly_();
  if (use_partial_assembly_)
  {
    a_->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
  }

  const bool use_device_qf = UseDeviceCoefficients_();
  using_device_coeffs_ = use_device_qf;
  if (use_device_qf)
  {
    EnsureDeviceQuadrature_();
  }

  mfem::Coefficient* invrho = nullptr;
  mfem::Coefficient* ck = nullptr;
  mfem::Coefficient* src = nullptr;
  if (use_device_qf)
  {
    invrho = invrho_qf_coef_.get();
    ck = ck_qf_coef_.get();
    src = src_qf_coef_.get();
  }
  else
  {
    invrho = invrho_coef_.get();
    ck = ck_coef_.get();
    src = src_coef_.get();
  }

  // Reaction/mass: (1/ρ σ, v)
  auto* mass_integ = new mfem::MassIntegrator(*invrho);
  if (use_device_qf && mass_qs_)
  {
    mass_integ->SetIntRule(&mass_qs_->GetIntRule(0));
  }
  a_->AddDomainIntegrator(mass_integ);

  // Volume diffusion term: ( (α/ρ) ∇σ, ∇v )
  auto* diff_integ = new mfem::DiffusionIntegrator(*ck);
  if (use_device_qf && sigma_qs_)
  {
    diff_integ->SetIntRule(&sigma_qs_->GetIntRule(0));
  }
  a_->AddDomainIntegrator(diff_integ);

  // If Σ-space is discontinuous, add IPDG face terms (interior only).
  const bool use_exact_face_operator = UseExactDGFaceOperator_();
  if (IsDGSpace(sfes_) && !use_exact_face_operator)
  {
    if (debug_qf && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=build-bilinear-dg-face"
                << " sigma_fqs=" << (sigma_fqs_ ? 1 : 0)
                << " ck_face_qf_coef=" << (ck_face_qf_coef_ ? 1 : 0)
                << std::endl;
    }
    const double sigma = static_cast<double>(ip_kind_); // -1(SIPG), 0, +1
    const double kappa = IPDGPenalty(sfes_);            // penalty scaling
    mfem::Coefficient* ck_face =
      use_device_qf ? ck_face_qf_coef_.get() : ck;
    auto* dg_diff = new mfem::DGDiffusionIntegrator(*ck_face, sigma, kappa);
    if (use_device_qf && sigma_fqs_)
    {
      dg_diff->SetIntRule(&sigma_fqs_->GetIntRule(0));
    }
    a_->AddInteriorFaceIntegrator(dg_diff);
    // For the IGR sigma equation, default behavior is natural boundary
    // conditions. Do not add DG boundary diffusion terms unless a specific
    // weak boundary closure is implemented.
  }
  else if (IsDGSpace(sfes_) && debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=build-bilinear-dg-face"
              << " mode=exact-side-qdata-wrapper"
              << " stock_mfem_face=0"
              << std::endl;
  }

  b_ = std::make_unique<mfem::ParLinearForm>(&sfes_);
  auto* rhs_integ = new mfem::DomainLFIntegrator(*src);
  if (use_device_qf && rhs_qs_)
  {
    rhs_integ->SetIntRule(&rhs_qs_->GetIntRule(0));
  }
  b_->AddDomainIntegrator(rhs_integ);
  b_->UseFastAssembly(use_partial_assembly_);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=build-bilinear-done" << "\n";
  }
}

bool IGRSolver::ShouldUsePartialAssembly_() const
{
  const bool device_backend =
    mfem::Device::Allows(mfem::Backend::DEVICE_MASK);
  if (device_backend)
  {
    const bool requested =
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_PARTIAL_ASSEMBLY", true);
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=partial-assembly-policy"
                << " requested=" << (requested ? "on" : "off")
                << " effective=on"
                << " reason=device-required-partial-assembly"
                << " precond=" << PrecondTypeName(prec_type_) << "\n";
    }
    return true;
  }

  bool use_partial =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_PARTIAL_ASSEMBLY", false);
  if (use_partial && PreconditionerRequiresAssembledMatrix(prec_type_))
  {
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=partial-assembly-policy"
                << " requested=on"
                << " effective=off"
                << " reason=matrix-preconditioner"
                << " precond=" << PrecondTypeName(prec_type_) << "\n";
    }
    use_partial = false;
  }
  return use_partial;
}

bool IGRSolver::UseExactDGFaceOperator_() const
{
  return exact_dg_face_operator_ ||
         ::dg_igr::EnvFlagEnabledDefault(
           "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR",
           false);
}

bool IGRSolver::ShouldUseDirectIdentitySystem_(bool use_device_qf,
                                               std::string& reason) const
{
  const bool requested =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_DIRECT_IDENTITY_SYSTEM",
                                    true);
  if (!requested)
  {
    reason = "disabled-by-env";
    return false;
  }
  if (!IsDGSpace(sfes_))
  {
    reason = "space-not-dg";
    return false;
  }
  if (!use_partial_assembly_)
  {
    reason = "partial-assembly-disabled";
    return false;
  }
  if (!use_device_qf)
  {
    reason = "device-qf-disabled";
    return false;
  }
  if (ess_tdof_list_.Size() != 0)
  {
    reason = "essential-bc";
    return false;
  }
  const mfem::Operator* prolongation = sfes_.GetProlongationMatrix();
  if (!mfem::IsIdentityProlongation(prolongation))
  {
    reason = "nonidentity-prolongation";
    return false;
  }
  if (sfes_.GetTrueVSize() != sfes_.GetVSize())
  {
    reason = "true-local-size-mismatch";
    return false;
  }
  if (!a_ || !b_)
  {
    reason = "forms-not-built";
    return false;
  }
  if (a_->Height() != sfes_.GetVSize() || b_->Size() != sfes_.GetVSize())
  {
    reason = "form-size-mismatch";
    return false;
  }
  reason = "active";
  return true;
}

mfem::Operator* IGRSolver::ActiveLinearOperator_() const
{
  if (direct_identity_system_)
  {
    if (Ah_.Ptr())
    {
      return Ah_.Ptr();
    }
    return a_.get();
  }
  return Ah_.Ptr();
}

void IGRSolver::AssembleSystem(const mfem::ParGridFunction& U)
{
  ::dg_igr::NVTXRange range("IGR/assemble-system");
  const auto assemble_start = std::chrono::steady_clock::now();

  // Needed for DG/IPDG assembly: variable coeffs evaluated on shared faces
  // will read U on neighbor elements.
  // For H1 sigma spaces, no DG/IP face terms are assembled, so this exchange is
  // unnecessary. In serial runs, neighbor exchange is also a no-op with extra
  // overhead; skip it in that case as well.
  if (IsDGSpace(sfes_) && mfem::Mpi::WorldSize() > 1)
  {
    ::dg_igr::NVTXRange exchange_range("IGR/state-face-neighbor-exchange");
    const_cast<mfem::ParGridFunction&>(U).ExchangeFaceNbrData();
  }
  MFEM_VERIFY(VectorIsFinite(U),
              "IGR state contains non-finite values before linear assembly.");

  const bool device_backend = mfem::Device::Allows(mfem::Backend::DEVICE_MASK);
  const bool use_device_qf = UseDeviceCoefficients_();
  const bool exact_face_requested = UseExactDGFaceOperator_();
  const bool is_dg_space = IsDGSpace(sfes_);
  const bool use_exact_face_operator = is_dg_space && exact_face_requested;
  const bool local_alpha =
    alpha_length_scale_policy_ == AlphaLengthScalePolicy::Local;
  last_exact_face_requested_ = exact_face_requested;
  last_exact_face_effective_ = false;
  last_exact_face_reason_ =
    exact_face_requested
      ? (is_dg_space ? "pending" : "space-not-dg")
      : "not-requested";
  last_exact_face_local_nf_ = 0;
  last_exact_face_shared_nf_ = 0;
  last_exact_face_operator_mult_calls_ = 0;
  const bool debug_qf =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=assemble-use-device-qf"
              << " enabled=" << (use_device_qf ? 1 : 0) << "\n";
  }
  if (use_exact_face_operator)
  {
    if (!use_device_qf)
    {
      last_exact_face_reason_ = "requires-device-qf";
      MFEM_VERIFY(false,
                  "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR requires "
                  "DG_IGR_IGR_DEVICE_QF=1.");
    }
    if (ess_bdr_.Size() != 0)
    {
      last_exact_face_reason_ = "requires-natural-boundary";
      MFEM_VERIFY(false,
                  "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR currently supports only "
                  "natural/no essential sigma boundary conditions.");
    }
  }

  if (use_device_qf != using_device_coeffs_)
  {
    BuildBilinearForm_();
  }

  if (use_device_qf)
  {
    ::dg_igr::NVTXRange coeff_range("IGR/device-coefficients");
    UpdateDeviceCoefficients_(U);
    // Keep the host coefficient views coherent for env-only assembled
    // preconditioner screens.  Their evaluation triggers a host read only if
    // such a screen is actually selected; the production device operator
    // continues to consume the QuadratureFunction coefficients above.
    invrho_coef_->SetState(U);
    invrho_coef_->SetRhoFloor(rho_floor_);
  }
  else
  {
    ::dg_igr::NVTXRange coeff_range("IGR/host-coefficients");
    // Ensure host-visible state for coefficient evaluation.
    // The IGR coefficients call GridFunction::GetVectorValue(), which is host-only.
    // On device backends, default to syncing unless explicitly disabled.
    const bool hostsync_default = device_backend;
    if (::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_FORCE_STATE_HOSTSYNC",
                                        hostsync_default))
    {
      (void)U.HostRead();
    }

    // Refresh coefficient views of the current state and params
    invrho_coef_->SetState(U);
    invrho_coef_->SetRhoFloor(rho_floor_);

    // Build gradient backend inputs
    if (backend_ == GradBackend::H1Projection)
    {
      EnsureH1VelocitySpace_();

      // Project velocity u = m/ρ into H1^dim
      VelocityFromConsCoefficient vel_coeff(dim_, &U, rho_floor_);
      vel_coeff.SetRhoFloor(rho_floor_);
      vel_coeff.SetState(&U);

      // DG state-derived velocity is generally discontinuous; use arithmetic
      // averaging at shared H1 dofs for partition-independent projection.
      vel_h1_->ProjectDiscCoefficient(vel_coeff, mfem::GridFunction::ARITHMETIC);

      // Tell RHS coefficient to use this continuous velocity field
      src_coef_->SetVelField(vel_h1_.get());
    }
    else
    {
      // BrokenDG backend: remove any projected velocity
      src_coef_->SetVelField(nullptr);
    }

    src_coef_->SetState(U);
    src_coef_->SetAlpha(alpha_);
    src_coef_->SetAlphaCoefficient(alpha_coef_.get());
    src_coef_->SetRhoFloor(rho_floor_);
  }

  // Keep MFEM form internals consistent before re-assembly across policy modes.
  a_->Update();
  b_->Update();
  b_->UseFastAssembly(use_partial_assembly_);

  // Re-assemble values (sparsity pattern unchanged)
  {
    ::dg_igr::NVTXRange assemble_form_range("IGR/mfem-form-assemble");
    a_->Assemble();
    b_->Assemble();
  }

  // Rebuild essential dofs if user set Dirichlet BCs (for H1 spaces).
  ess_tdof_list_.SetSize(0);
  if (ess_bdr_.Size() > 0)
  {
    mfem::Array<int> dbdr(ess_bdr_);
    sfes_.GetEssentialTrueDofs(dbdr, ess_tdof_list_);
  }

  const auto assemble_stop = std::chrono::steady_clock::now();
  last_assemble_seconds_ =
    std::chrono::duration<double>(assemble_stop - assemble_start).count();
  total_assemble_seconds_ += last_assemble_seconds_;

  // Rebuild the system operator; clear handle first.
  Ah_.Clear();
  Ah_.SetType(use_partial_assembly_ ? mfem::Operator::ANY_TYPE
                                    : mfem::Operator::Hypre_ParCSR);
  direct_identity_system_ = false;
  last_direct_identity_reason_ = "not-evaluated";

  // Warm-start: FormLinearSystem pulls current sigma_ into X_
  const auto form_start = std::chrono::steady_clock::now();
  std::string direct_identity_reason;
  {
    ::dg_igr::NVTXRange form_range("IGR/form-linear-system");
    if (ShouldUseDirectIdentitySystem_(use_device_qf, direct_identity_reason))
    {
      X_.MakeRef(sigma_, 0, sigma_.Size());
      B_.MakeRef(*b_, 0, b_->Size());
      direct_identity_system_ = true;
      ++direct_identity_system_uses_;
    }
    else
    {
      a_->FormLinearSystem(ess_tdof_list_, sigma_, *b_, Ah_, X_, B_);
    }
  }
  last_direct_identity_reason_ = direct_identity_reason;

  if (use_exact_face_operator)
  {
    ::dg_igr::NVTXRange exact_setup_range("IGR/exact-face-setup");
    if (!use_partial_assembly_)
    {
      last_exact_face_reason_ = "requires-partial-assembly";
      MFEM_VERIFY(false,
                  "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR requires partial assembly.");
    }
    mfem::Mesh* mesh = sfes_.GetMesh();
    if (!mesh)
    {
      last_exact_face_reason_ = "requires-valid-mesh";
      MFEM_VERIFY(false,
                  "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR requires a valid mesh.");
    }
    auto* pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
    const mfem::Geometry::Type face_geom = mesh->GetTypicalFaceGeometry();
    const int exact_face_order =
      2 * std::max(SpaceOrder(sfes_), SpaceOrder(vfes_));
    const mfem::IntegrationRule& exact_face_ir =
      mfem::IntRules.Get(face_geom, exact_face_order);
    mfem::FaceQuadratureSpace exact_fqs(*mesh,
                                        exact_face_ir,
                                        mfem::FaceType::Interior);
    std::string reason;
    if (!dg_face::SupportsInteriorFaceSideDensityProjection2D(vfes_,
                                                              exact_fqs,
                                                              &reason))
    {
      last_exact_face_reason_ = "unsupported-interior-side-density";
      MFEM_VERIFY(false,
                  "Unsupported exact IGR side-density face operator: "
                    << reason);
    }
    if (!dg_face::SupportsInteriorFaceScalarTraceProjection2D(sfes_,
                                                              exact_fqs,
                                                              &reason))
    {
      last_exact_face_reason_ = "unsupported-interior-sigma-trace";
      MFEM_VERIFY(false,
                  "Unsupported exact IGR sigma face operator: " << reason);
    }

    mfem::Vector side_q;
    dg_face::InteriorFaceSideDensityProjectionScratch side_scratch;
    if (local_alpha)
    {
      MFEM_VERIFY(device_element_alpha_.Size() == vfes_.GetNE(),
                  "Exact local-alpha face projection requires the current "
                  "device element-alpha cache.");
    }
    const auto side_info = local_alpha
      ? dg_face::ProjectInteriorFaceDensityCoefficient2D(vfes_,
                                                         exact_fqs,
                                                         U,
                                                         device_element_alpha_,
                                                         rho_floor_,
                                                         side_q,
                                                         &side_scratch)
      : dg_face::ProjectInteriorFaceDensityCoefficient2D(vfes_,
                                                         exact_fqs,
                                                         U,
                                                         alpha_,
                                                         rho_floor_,
                                                         side_q,
                                                         &side_scratch);
    last_exact_face_local_nf_ = side_info.nf;
    const bool ingestion_audit =
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_INGESTION_AUDIT", false);
    if (ingestion_audit)
    {
      LogFaceCoefficientAudit("local",
                              side_q,
                              alpha_,
                              rho_floor_,
                              linear_solve_calls_ + 1,
                              vfes_.GetComm(),
                              local_alpha
                                ? &side_scratch.side_alpha
                                : nullptr,
                              side_info.nq);
    }

    mfem::Vector shared_side_q;
    mfem::Vector shared_face_side_alpha;
    dg_face::SharedFaceSideDensityProjectionScratch shared_side_scratch;
    dg_face::SharedFaceSideDensityProjectionInfo shared_side_info;
    if (pmesh && pmesh->GetNSharedFaces() > 0)
    {
      if (!dg_face::SupportsSharedFaceSideDensityProjection2D(vfes_,
                                                              exact_face_ir,
                                                              &reason))
      {
        last_exact_face_reason_ = "unsupported-shared-side-density";
        MFEM_VERIFY(false,
                    "Unsupported exact IGR shared side-density face operator: "
                      << reason);
      }
      if (!dg_face::SupportsSharedFaceDGDiffusionApplyData2D(sfes_,
                                                             exact_face_ir,
                                                             &reason))
      {
        last_exact_face_reason_ = "unsupported-shared-sigma-trace";
        MFEM_VERIFY(false,
                    "Unsupported exact IGR shared sigma face operator: "
                      << reason);
      }
      if (local_alpha)
      {
        BuildSharedFaceSideAlpha(vfes_, alpha_, shared_face_side_alpha);
      }
      shared_side_info = local_alpha
        ? dg_face::ProjectSharedFaceDensityCoefficient2D(
            vfes_,
            exact_face_ir,
            U,
            shared_face_side_alpha,
            rho_floor_,
            shared_side_q,
            &shared_side_scratch)
        : dg_face::ProjectSharedFaceDensityCoefficient2D(vfes_,
                                                         exact_face_ir,
                                                         U,
                                                         alpha_,
                                                         rho_floor_,
                                                         shared_side_q,
                                                         &shared_side_scratch);
      last_exact_face_shared_nf_ = shared_side_info.nf;
      if (ingestion_audit)
      {
        LogFaceCoefficientAudit("shared",
                                shared_side_q,
                                alpha_,
                                rho_floor_,
                                linear_solve_calls_ + 1,
                                vfes_.GetComm(),
                                local_alpha
                                  ? &shared_side_scratch.side_alpha
                                  : nullptr,
                                shared_side_info.nq);
      }
    }

    mfem::Operator* base_op =
      direct_identity_system_ ? static_cast<mfem::Operator*>(a_.get())
                              : Ah_.Ptr();
    MFEM_VERIFY(base_op != nullptr,
                "Exact IGR face operator requires formed base operator.");
    ExactL2DGFaceOperator* face_op = nullptr;
    const mfem::real_t face_kappa =
      static_cast<mfem::real_t>(IPDGPenalty(sfes_));
    const dg_face::DGDiffusionFaceScaling face_scaling =
      ReadIPDGFaceScaling();
    face_op = dynamic_cast<ExactL2DGFaceOperator*>(
      exact_face_operator_.get());
    const bool reused_face_geometry =
      face_op && face_op->CanRefresh(sfes_,
                                     side_q,
                                     shared_side_q,
                                     face_kappa,
                                     face_scaling);
    if (reused_face_geometry)
    {
      face_op->RefreshCoefficients(side_q, shared_side_q);
    }
    else
    {
      exact_face_operator_.reset();
      exact_face_operator_ = std::make_unique<ExactL2DGFaceOperator>(
        sfes_,
        exact_fqs,
        side_q,
        exact_face_ir,
        shared_side_q,
        static_cast<mfem::real_t>(ip_kind_),
        face_kappa,
        face_scaling);
      face_op = dynamic_cast<ExactL2DGFaceOperator*>(
        exact_face_operator_.get());
    }
    MFEM_VERIFY(face_op != nullptr,
                "Exact DG face operator construction failed.");
    if (!direct_identity_system_)
    {
      Ah_.SetOperatorOwner(false);
    }
    Ah_.Reset(new ExactFaceSumOperator(base_op,
                                       face_op,
                                       !direct_identity_system_,
                                       false));
    last_exact_face_effective_ = true;
    last_exact_face_reason_ = "active";

    if (debug_qf && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=exact-dg-face-operator"
                << " enabled=1"
                << " nf=" << side_info.nf
                << " nq=" << side_info.nq
                << " qf_size=" << side_info.qf_size
                << " shared_nf=" << shared_side_info.nf
                << " shared_qf_size=" << shared_side_info.qf_size
                << " face_scaling=" << IPDGFaceScalingName(face_scaling)
                << " geometry_reused=" << (reused_face_geometry ? 1 : 0)
                << " mode=local-and-shared-2d"
                << std::endl;
    }
  }
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=exact-dg-face-policy"
              << " requested=" << (last_exact_face_requested_ ? "on" : "off")
              << " effective=" << (last_exact_face_effective_ ? "on" : "off")
              << " reason=" << LogToken(last_exact_face_reason_)
              << " local_nf=" << last_exact_face_local_nf_
              << " shared_nf=" << last_exact_face_shared_nf_
              << " operator_uses=" << exact_face_operator_uses_
              << std::endl;
  }
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=direct-identity-system"
              << " requested="
              << (::dg_igr::EnvFlagEnabledDefault(
                    "DG_IGR_IGR_DIRECT_IDENTITY_SYSTEM",
                    true) ? "on" : "off")
              << " effective=" << (direct_identity_system_ ? "on" : "off")
              << " reason=" << LogToken(last_direct_identity_reason_)
              << " uses=" << direct_identity_system_uses_
              << std::endl;
  }
  const auto form_stop = std::chrono::steady_clock::now();
  last_form_system_seconds_ =
    std::chrono::duration<double>(form_stop - form_start).count();
  total_form_system_seconds_ += last_form_system_seconds_;
}

void IGRSolver::DoSmootherSweeps(int sweeps, bool /*warm_start*/)
{
  ::dg_igr::NVTXRange range("IGR/smoother-sweeps");
  mfem::Operator* A_op = ActiveLinearOperator_();
  MFEM_VERIFY(A_op != nullptr,
              "IGR: linear operator is not assembled before smoother sweeps.");
  mfem::HypreParMatrix* A_hypre =
    direct_identity_system_ ? nullptr : Ah_.Is<mfem::HypreParMatrix>();

  std::unique_ptr<mfem::Solver> smoother;
  if (A_hypre)
  {
    auto hypre_sm = std::make_unique<mfem::HypreSmoother>(*A_hypre);
    switch (smoot_)
    {
    case SmootherType::Jacobi:
      hypre_sm->SetType(mfem::HypreSmoother::Jacobi);
      break;
    case SmootherType::L1Jacobi:
      hypre_sm->SetType(mfem::HypreSmoother::l1Jacobi);
      break;
    case SmootherType::L1GS:
      hypre_sm->SetType(mfem::HypreSmoother::l1GS);
      break;
    case SmootherType::L1GStr:
      hypre_sm->SetType(mfem::HypreSmoother::l1GStr);
      break;
    case SmootherType::GS:
      hypre_sm->SetType(mfem::HypreSmoother::GS);
      break;
    }
    hypre_sm->SetPositiveDiagonal(true);
    smoother = std::move(hypre_sm);
  }
  else
  {
    smoother = MakePositiveOperatorJacobi(*a_, ess_tdof_list_);
  }

  mfem::Vector r(B_.Size()), z(X_.Size());
  for (int it = 0; it < sweeps; ++it)
  {
    // r = b - A x
    r = B_;
    A_op->Mult(X_, r);
    r.Neg();
    r += B_;

    // z ≈ M^{-1} r  (one application of the smoother/preconditioner)
    z = 0.0;
    smoother->Mult(r, z);

    // x <- x + z  (Richardson step)
    X_ += z;
  }

  if (direct_identity_system_)
  {
    X_.SyncAliasMemory(sigma_);
  }
  else
  {
    a_->RecoverFEMSolution(X_, *b_, sigma_);
  }
}

void IGRSolver::SolveEllipticSystem_(bool warm_start)
{
  ::dg_igr::NVTXRange range("IGR/solve-elliptic-system");
  mfem::Operator* A_op = ActiveLinearOperator_();
  MFEM_VERIFY(A_op != nullptr,
              "IGR: linear operator is not assembled before Krylov solve.");
  mfem::HypreParMatrix* A_hypre =
    direct_identity_system_ ? nullptr : Ah_.Is<mfem::HypreParMatrix>();
  const bool has_hypre_matrix = (A_hypre != nullptr);
  auto* exact_face_sum_op = dynamic_cast<ExactFaceSumOperator*>(A_op);

  MPI_Comm comm = sfes_.GetComm();
  const int operator_probe_every =
    ReadEnvInt("DG_IGR_IGR_OPERATOR_PROBE_EVERY", 0);
  const int operator_probe_start =
    ReadEnvInt("DG_IGR_IGR_OPERATOR_PROBE_START", 1);
  const int solve_call = linear_solve_calls_ + 1;
  const int event_log_min_iters =
    ReadEnvInt("DG_IGR_IGR_EVENT_LOG_MIN_ITERS", 0);
  const char* event_capture_prefix_text =
    std::getenv("DG_IGR_IGR_EVENT_CAPTURE_PREFIX");
  const bool event_capture_enabled =
    event_capture_prefix_text && event_capture_prefix_text[0] != '\0';
  const int event_capture_min_iters =
    ReadEnvInt("DG_IGR_IGR_EVENT_CAPTURE_MIN_ITERS",
               event_log_min_iters > 0 ? event_log_min_iters : lin_maxit_);
  const bool true_residual_requested =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_TRUE_RESIDUAL", false);
  const bool true_initial_residual_requested =
    true_residual_requested && ::dg_igr::EnvFlagEnabledDefault(
                                 "DG_IGR_IGR_TRUE_INITIAL_RESIDUAL", false);
  const int extended_certificate_start =
    ReadEnvInt("DG_IGR_IGR_EXTENDED_CERTIFICATE_START", 1000);
  const double extended_certificate_factor =
    ReadEnvReal("DG_IGR_IGR_EXTENDED_CERTIFICATE_FACTOR", 5.0);
  MFEM_VERIFY(extended_certificate_start > 0,
              "DG_IGR_IGR_EXTENDED_CERTIFICATE_START must be positive.");
  MFEM_VERIFY(extended_certificate_factor > 0.0,
              "DG_IGR_IGR_EXTENDED_CERTIFICATE_FACTOR must be positive.");
  const bool extended_certificate_enabled =
    ::dg_igr::EnvFlagEnabledDefault(
      "DG_IGR_IGR_EXTENDED_CERTIFICATE", false) &&
    exact_face_sum_op && lin_maxit_ > extended_certificate_start;
  const bool run_operator_probe =
    operator_probe_every > 0 && solve_call >= operator_probe_start &&
    (solve_call - operator_probe_start) % operator_probe_every == 0;
  const int operator_probe_steps =
    ReadEnvInt("DG_IGR_IGR_OPERATOR_PROBE_STEPS", 12);
  if (run_operator_probe)
  {
    LogOperatorAlgebraProbe(
      "full", solve_call, *A_op, comm, operator_probe_steps);
    if (exact_face_sum_op)
    {
      LogOperatorAlgebraProbe("base",
                              solve_call,
                              exact_face_sum_op->BaseOperator(),
                              comm,
                              operator_probe_steps);
      LogOperatorAlgebraProbe("face",
                              solve_call,
                              exact_face_sum_op->FaceOperator(),
                              comm,
                              operator_probe_steps);
    }
  }
  const bool is_dg = IsDGSpace(sfes_);
  const bool spd_ok = IsSPD_DefaultAssumption(sfes_, ip_kind_);
  const bool using_cuda = IsCudaBackendActive();
  const bool prev_linear_converged = last_linear_converged_;
  const double prev_linear_final_norm = last_linear_final_norm_;

  last_linear_iters_ = 0;
  last_linear_converged_ = false;
  last_linear_final_norm_ = std::numeric_limits<double>::quiet_NaN();
  last_linear_initial_norm_ = std::numeric_limits<double>::quiet_NaN();
  last_linear_true_residual_abs_ =
    std::numeric_limits<double>::quiet_NaN();
  last_linear_true_residual_rel_rhs_ =
    std::numeric_limits<double>::quiet_NaN();
  last_independent_certificate_relative_limit_ =
    std::numeric_limits<double>::quiet_NaN();
  last_independent_certificate_absolute_limit_ =
    std::numeric_limits<double>::quiet_NaN();
  last_linear_true_initial_residual_abs_ =
    std::numeric_limits<double>::quiet_NaN();
  last_linear_true_residual_rel_initial_ =
    std::numeric_limits<double>::quiet_NaN();
  last_linear_recomputed_preconditioned_initial_norm_ =
    std::numeric_limits<double>::quiet_NaN();
  last_linear_recomputed_preconditioned_final_norm_ =
    std::numeric_limits<double>::quiet_NaN();
  last_linear_solve_seconds_ = 0.0;
  last_pc_setup_seconds_ = 0.0;
  last_chebyshev_diagonal_seconds_ = 0.0;
  last_chebyshev_setup_seconds_ = 0.0;
  last_chebyshev_eigenvalue_reused_ = false;
  last_chebyshev_smoother_reused_ = false;
  last_chebyshev_eigenvalue_estimate_ = 0.0;
  last_krylov_operator_apply_seconds_ = 0.0;
  last_krylov_preconditioner_apply_seconds_ = 0.0;
  last_krylov_preconditioner_operator_apply_seconds_ = 0.0;
  last_krylov_operator_apply_count_ = 0;
  last_krylov_preconditioner_apply_count_ = 0;
  last_krylov_preconditioner_operator_apply_count_ = 0;
  last_linear_solver_name_ = "none";
  last_precond_name_ = "none";

  // --- Choose solver (Auto policy is space-aware) ---
  const LinearSolverType requested_solver = lin_solver_type_;
  LinearSolverType solver_choice = lin_solver_type_;
  if (solver_choice == LinearSolverType::Auto)
  {
    solver_choice = is_dg ? LinearSolverType::GMRES : LinearSolverType::CG;
  }

  // Guardrail: do not use SPD-only Krylov methods when SPD assumptions fail.
  if (RequiresSPD(solver_choice) && !spd_ok)
  {
    const LinearSolverType prev = solver_choice;
    solver_choice = LinearSolverType::GMRES;
    LogIGRFallback("operator-not-spd",
                   prev,
                   solver_choice,
                   prec_type_,
                   prec_type_);
  }

  // --- Choose preconditioner (Auto policy is space-aware) ---
  PrecondType prec_choice = prec_type_;
  if (prec_choice == PrecondType::Auto)
  {
    // CPU: keep BoomerAMG as robust default.
    // CUDA: prefer HypreSmoother for lower setup/solve overhead on current
    // inoue-style runs; fallback policy remains active for unsupported paths.
    (void)is_dg;
    prec_choice =
      using_cuda ? PrecondType::HypreSmoother : PrecondType::BoomerAMG;
  }

  // Guardrail: CG/MINRES should use symmetric/no preconditioners.
  if (RequiresSPD(solver_choice) && !IsSymmetricPC(prec_choice))
  {
    if (using_cuda)
    {
      MFEM_VERIFY(false,
                  "IGR device-closure requires symmetric preconditioning for "
                  "SPD Krylov solvers; requested preconditioner is incompatible.");
    }
    const PrecondType prev = prec_choice;
    prec_choice = PrecondType::BoomerAMG;
    LogIGRFallback("spd-solver-with-nonsymmetric-preconditioner",
                   requested_solver,
                   solver_choice,
                   prev,
                   prec_choice);
  }

  if (using_cuda)
  {
    MFEM_VERIFY(!has_hypre_matrix,
                "IGR device-closure requires partial-assembly operator path "
                "(no assembled Hypre matrix).");
    MFEM_VERIFY(prec_choice == PrecondType::HypreSmoother ||
                  prec_choice == PrecondType::OperatorChebyshev ||
                  prec_choice == PrecondType::None,
                "IGR device-closure on CUDA supports only "
                "--igr-preconditioner hypresmoother|chebyshev|none.");
  }
  else if (!has_hypre_matrix &&
           (prec_choice == PrecondType::BoomerAMG ||
            prec_choice == PrecondType::HypreILU ||
            prec_choice == PrecondType::HypreEuclid ||
            prec_choice == PrecondType::ParaSails))
  {
    const PrecondType prev = prec_choice;
    prec_choice = PrecondType::HypreSmoother;
    LogIGRFallback("operator-path-without-hypre-matrix",
                   requested_solver,
                   solver_choice,
                   prev,
                   prec_choice);
  }

  KrylovApplyTelemetry krylov_telemetry;
  krylov_telemetry.record_seconds =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_KRYLOV_APPLY_TIMING",
                                    false);
  krylov_telemetry.sync =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_KRYLOV_APPLY_SYNC",
                                    false);
  const bool use_krylov_telemetry =
    krylov_telemetry.record_seconds ||
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_KRYLOV_APPLY_COUNTS",
                                    true);

  const auto pc_setup_start = std::chrono::steady_clock::now();
  std::unique_ptr<KrylovTelemetryOperator> cheb_operator_telemetry;
  std::unique_ptr<mfem::Solver> owned_prec;
  mfem::Solver* prec = nullptr;
  auto set_owned_preconditioner =
    [&](std::unique_ptr<mfem::Solver> candidate)
    {
      owned_prec = std::move(candidate);
      prec = owned_prec.get();
    };
  bool using_operator_jacobi = false;
  bool using_operator_jacobi_exact_face_diag = false;
  bool using_operator_chebyshev = false;
  bool using_operator_chebyshev_base = false;
  bool using_static_dg_lor_amg = false;
  bool static_dg_lor_amg_reused = false;
  std::string static_dg_lor_amg_reason = "not-requested";
  double static_dg_lor_amg_setup_seconds = 0.0;
  double static_dg_lor_alpha_proxy =
    std::numeric_limits<double>::quiet_NaN();
  const bool static_dg_lor_amg_requested =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_STATIC_DG_LOR_AMG", false);
  bool using_exact_element_block_jacobi = false;
  const bool exact_element_block_jacobi_env_requested =
    ::dg_igr::EnvFlagEnabledDefault(
      "DG_IGR_IGR_EXACT_ELEMENT_BLOCK_JACOBI", false);
  const bool exact_face_pair_schwarz_requested =
    ::dg_igr::EnvFlagEnabledDefault(
      "DG_IGR_IGR_EXACT_FACE_PAIR_SCHWARZ", false);
  const bool exact_element_block_jacobi_requested =
    exact_element_block_jacobi_env_requested ||
    exact_face_pair_schwarz_requested;
  const bool exact_element_block_step_reuse_requested =
    ::dg_igr::EnvFlagEnabledDefault(
      "DG_IGR_IGR_EXACT_ELEMENT_BLOCK_STEP_REUSE", false);
  const bool shared_correlation_block_requested =
    ::dg_igr::EnvFlagEnabledDefault(
      "DG_IGR_IGR_SHARED_CORRELATION_BLOCK", false);
  const bool original_dg0_two_level_requested =
    ::dg_igr::EnvFlagEnabledDefault(
      "DG_IGR_IGR_ORIGINAL_DG0_TWO_LEVEL", false);
  MFEM_VERIFY(!original_dg0_two_level_requested ||
                (!static_dg_lor_amg_requested &&
                 exact_element_block_jacobi_env_requested &&
                 !exact_face_pair_schwarz_requested),
              "DG_IGR_IGR_ORIGINAL_DG0_TWO_LEVEL requires exact element "
              "blocks and excludes the refined LOR prototype.");
  MFEM_VERIFY(!exact_element_block_step_reuse_requested ||
                (exact_element_block_jacobi_requested &&
                 !static_dg_lor_amg_requested &&
                 !original_dg0_two_level_requested),
              "DG_IGR_IGR_EXACT_ELEMENT_BLOCK_STEP_REUSE requires exact "
              "local blocks and excludes the LOR/DG0 prototypes.");
  MFEM_VERIFY(!shared_correlation_block_requested ||
                (exact_element_block_jacobi_env_requested &&
                 !exact_face_pair_schwarz_requested),
              "DG_IGR_IGR_SHARED_CORRELATION_BLOCK requires exact element "
              "blocks and excludes the face-pair Schwarz screen.");
  MFEM_VERIFY(!exact_face_pair_schwarz_requested ||
                (!static_dg_lor_amg_requested &&
                 !original_dg0_two_level_requested &&
                 !shared_correlation_block_requested),
              "DG_IGR_IGR_EXACT_FACE_PAIR_SCHWARZ is an isolated exact "
              "local-patch screen and excludes the LOR, DG0, and shared-"
              "correlation prototypes.");
  bool using_original_dg0_two_level = false;
  ExactElementBlockJacobiBuild exact_element_block_build;
  std::unique_ptr<mfem::Solver> exact_element_block_prec;
  mfem::Solver* cached_exact_element_block_prec = nullptr;
  bool exact_element_block_cache_reused = false;
  double exact_element_block_setup_seconds = 0.0;
  {
    ::dg_igr::NVTXRange pc_range("IGR/preconditioner-setup");
    if (static_dg_lor_amg_requested)
    {
      const auto lor_setup_start = Clock::now();
      mfem::Mesh* mesh = sfes_.GetMesh();
      const auto* l2_collection =
        dynamic_cast<const mfem::L2_FECollection*>(sfes_.FEColl());
      const int order = sfes_.GetNE() > 0 ? SpaceOrder(sfes_) : -1;
      const double penalty = order >= 0
        ? IPDGPenalty(sfes_)
        : std::numeric_limits<double>::quiet_NaN();
      if (!is_dg || sfes_.GetVDim() != 1 ||
          l2_collection == nullptr)
      {
        static_dg_lor_amg_reason = "requires-scalar-l2";
      }
      else if (ip_kind_ != IPKind::SIPG || !spd_ok)
      {
        static_dg_lor_amg_reason = "requires-sipg";
      }
      else if (!exact_face_sum_op || !last_exact_face_effective_)
      {
        static_dg_lor_amg_reason = "requires-exact-face-operator";
      }
      else if (!direct_identity_system_)
      {
        static_dg_lor_amg_reason = "requires-direct-identity-system";
      }
      else if (ess_bdr_.Size() != 0 || ess_tdof_list_.Size() != 0)
      {
        static_dg_lor_amg_reason = "requires-natural-boundary";
      }
      else if (!mesh)
      {
        static_dg_lor_amg_reason = "requires-valid-mesh";
      }
      else if (mesh->Nonconforming())
      {
        static_dg_lor_amg_reason = "requires-conforming-mesh";
      }
      else if (mesh->Dimension() != 2 ||
               mesh->GetNumGeometries(mesh->Dimension()) != 1 ||
               mesh->GetTypicalElementGeometry() != mfem::Geometry::SQUARE ||
               !mfem::UsesTensorBasis(sfes_))
      {
        static_dg_lor_amg_reason = "requires-tensor-product-quads";
      }
      else if (sfes_.IsVariableOrder() || order < 1 || order > 3)
      {
        static_dg_lor_amg_reason = "requires-uniform-p1-p3";
      }
      else if (!alpha_coef_ || !std::isfinite(alpha_) || alpha_ <= 0.0)
      {
        static_dg_lor_amg_reason = "requires-positive-finite-alpha";
      }
      else
      {
        const bool cache_matches =
          static_dg_lor_amg_ && static_dg_lor_discretization_ &&
          static_dg_lor_form_ && static_dg_lor_mass_coef_ &&
          static_dg_lor_diffusion_coef_ &&
          static_dg_lor_diagonal_.Size() == A_op->Height() &&
          static_dg_lor_mesh_ == mesh &&
          static_dg_lor_mesh_sequence_ == mesh->GetSequence() &&
          static_dg_lor_sfes_sequence_ == sfes_.GetSequence() &&
          static_dg_lor_alpha_ == alpha_ &&
          static_dg_lor_alpha_policy_ == alpha_length_scale_policy_ &&
          static_dg_lor_penalty_ == penalty &&
          static_dg_lor_order_ == order &&
          static_dg_lor_amg_coarsen_type_ == amg_coarsen_type_ &&
          static_dg_lor_amg_relax_type_ == amg_relax_type_ &&
          static_dg_lor_amg_max_levels_ == amg_max_levels_ &&
          static_dg_lor_amg_strong_threshold_ == amg_strong_threshold_ &&
          static_dg_lor_amg_->Height() == A_op->Height() &&
          static_dg_lor_amg_->Width() == A_op->Width();
        if (cache_matches)
        {
          prec = static_dg_lor_amg_.get();
          using_static_dg_lor_amg = true;
          static_dg_lor_amg_reused = true;
          static_dg_lor_amg_reason = "active";
          static_dg_lor_alpha_proxy = static_dg_lor_alpha_proxy_;
        }
        else
        {
          ::dg_igr::NVTXRange lor_range("IGR/static-dg-lor-amg-setup");
          // Tear down in dependency order: solver -> LOR -> borrowed form ->
          // coefficient.  The new objects are committed only after dimensions
          // are checked against the unchanged exact solve operator.
          static_dg_lor_amg_.reset();
          static_dg_lor_discretization_.reset();
          static_dg_lor_form_.reset();
          static_dg_lor_diffusion_coef_.reset();
          static_dg_lor_mass_coef_.reset();
          static_dg_lor_diagonal_.SetSize(0);

          static_dg_lor_alpha_proxy = alpha_;
          if (alpha_length_scale_policy_ == AlphaLengthScalePolicy::Local)
          {
            MFEM_VERIFY(device_element_alpha_.Size() == sfes_.GetNE(),
                        "Static DG LOR/AMG local-alpha proxy requires the "
                        "current element-alpha cache.");
            const mfem::real_t* element_alpha =
              device_element_alpha_.HostRead();
            double local_alpha_sum = 0.0;
            long long local_alpha_count = device_element_alpha_.Size();
            for (int element = 0; element < device_element_alpha_.Size();
                 ++element)
            {
              MFEM_VERIFY(std::isfinite(element_alpha[element]) &&
                            element_alpha[element] > 0.0,
                          "Static DG LOR/AMG received invalid local alpha.");
              local_alpha_sum += element_alpha[element];
            }
            double global_alpha_sum = 0.0;
            long long global_alpha_count = 0;
            MPI_Allreduce(&local_alpha_sum,
                          &global_alpha_sum,
                          1,
                          MPI_DOUBLE,
                          MPI_SUM,
                          comm);
            MPI_Allreduce(&local_alpha_count,
                          &global_alpha_count,
                          1,
                          MPI_LONG_LONG,
                          MPI_SUM,
                          comm);
            MFEM_VERIFY(global_alpha_count > 0 &&
                          std::isfinite(global_alpha_sum) &&
                          global_alpha_sum > 0.0,
                        "Static DG LOR/AMG could not form a positive global "
                        "local-alpha proxy.");
            static_dg_lor_alpha_proxy =
              global_alpha_sum / static_cast<double>(global_alpha_count);
          }
          MFEM_VERIFY(std::isfinite(static_dg_lor_alpha_proxy) &&
                        static_dg_lor_alpha_proxy > 0.0,
                      "Static DG LOR/AMG alpha proxy must be positive and "
                      "finite.");
          auto mass_coef =
            std::make_unique<mfem::ConstantCoefficient>(1.0);
          auto diffusion_coef =
            std::make_unique<mfem::ConstantCoefficient>(
              static_dg_lor_alpha_proxy);
          auto surrogate =
            std::make_unique<mfem::ParBilinearForm>(&sfes_);
          surrogate->AddDomainIntegrator(
            new mfem::MassIntegrator(*mass_coef));
          surrogate->AddDomainIntegrator(
            new mfem::DiffusionIntegrator(*diffusion_coef));
          surrogate->AddInteriorFaceIntegrator(
            new mfem::DGDiffusionIntegrator(*diffusion_coef,
                                            -1.0,
                                            penalty));

          auto lor = std::make_unique<mfem::ParLORDiscretization>(
            sfes_, mfem::BasisType::GaussLobatto);
          // Pinned MFEM's batched DG-LOR path inserts a physical-boundary
          // penalty even when the high-order form has natural boundaries.
          // Explicit legacy assembly honors the form's missing boundary-face
          // integrator and therefore preserves the intended surrogate BC.
          lor->LegacyAssembleSystem(*surrogate, ess_tdof_list_);
          auto lor_amg =
            std::make_unique<mfem::LORSolver<mfem::HypreBoomerAMG>>(*lor);
          mfem::Vector lor_diagonal(A_op->Height());
          lor->GetAssembledMatrix().GetDiag(lor_diagonal);
          mfem::HypreBoomerAMG& amg = lor_amg->GetSolver();
          amg.SetPrintLevel(0);
          amg.SetMaxIter(1);
          amg.SetTol(0.0);
          if (amg_coarsen_type_ >= 0)
          {
            amg.SetCoarsening(amg_coarsen_type_);
          }
          if (amg_relax_type_ >= 0)
          {
            amg.SetRelaxType(amg_relax_type_);
          }
          if (amg_max_levels_ >= 0)
          {
            amg.SetMaxLevels(amg_max_levels_);
          }
          if (amg_strong_threshold_ >= 0.0)
          {
            amg.SetStrengthThresh(amg_strong_threshold_);
          }

          if (lor_amg->Height() != A_op->Height() ||
              lor_amg->Width() != A_op->Width())
          {
            static_dg_lor_amg_reason = "lor-operator-size-mismatch";
          }
          else
          {
            static_dg_lor_mass_coef_ = std::move(mass_coef);
            static_dg_lor_diffusion_coef_ = std::move(diffusion_coef);
            static_dg_lor_form_ = std::move(surrogate);
            static_dg_lor_discretization_ = std::move(lor);
            static_dg_lor_amg_ = std::move(lor_amg);
            static_dg_lor_diagonal_ = lor_diagonal;
            static_dg_lor_diagonal_.UseDevice(true);
            static_dg_lor_mesh_ = mesh;
            static_dg_lor_mesh_sequence_ = mesh->GetSequence();
            static_dg_lor_sfes_sequence_ = sfes_.GetSequence();
            static_dg_lor_alpha_ = alpha_;
            static_dg_lor_alpha_proxy_ = static_dg_lor_alpha_proxy;
            static_dg_lor_alpha_policy_ = alpha_length_scale_policy_;
            static_dg_lor_penalty_ = penalty;
            static_dg_lor_order_ = order;
            static_dg_lor_amg_coarsen_type_ = amg_coarsen_type_;
            static_dg_lor_amg_relax_type_ = amg_relax_type_;
            static_dg_lor_amg_max_levels_ = amg_max_levels_;
            static_dg_lor_amg_strong_threshold_ = amg_strong_threshold_;
            prec = static_dg_lor_amg_.get();
            using_static_dg_lor_amg = true;
            static_dg_lor_amg_reason = "active";
          }
        }
      }
      static_dg_lor_amg_setup_seconds = ElapsedSeconds(lor_setup_start);
      if (mfem::Mpi::Root())
      {
        const char* alpha_policy = "unknown";
        switch (alpha_length_scale_policy_)
        {
        case AlphaLengthScalePolicy::Min: alpha_policy = "min"; break;
        case AlphaLengthScalePolicy::Average:
          alpha_policy = "average";
          break;
        case AlphaLengthScalePolicy::Max: alpha_policy = "max"; break;
        case AlphaLengthScalePolicy::Local: alpha_policy = "local"; break;
        }
        std::cout << "[igr-linear] event=static-dg-lor-amg"
                  << " solve_call=" << solve_call
                  << " requested=1"
                  << " effective=" << (using_static_dg_lor_amg ? 1 : 0)
                  << " reason=" << static_dg_lor_amg_reason
                  << " cache="
                  << (using_static_dg_lor_amg
                        ? (static_dg_lor_amg_reused ? "reused" : "built")
                        : "none")
                  << " setup_s=" << static_dg_lor_amg_setup_seconds
                  << " order=" << order
                  << " penalty=" << penalty
                  << " alpha=" << alpha_
                  << " alpha_proxy=" << static_dg_lor_alpha_proxy
                  << " alpha_policy=" << alpha_policy
                  << " surrogate_alpha=constant-global-arithmetic-mean"
                  << " exact_operator_local_alpha="
                  << (alpha_length_scale_policy_ ==
                        AlphaLengthScalePolicy::Local
                        ? 1
                        : 0)
                  << " assembly=legacy-natural"
                  << " high_order_basis="
                  << (l2_collection
                        ? mfem::BasisType::Name(
                            l2_collection->GetBasisType())
                        : "unknown")
                  << " lor_identity_mapping=prototype-unqualified"
                  << std::endl;
      }
    }

    if (exact_element_block_jacobi_requested)
    {
      const auto block_setup_start = Clock::now();
      if (prec_choice != PrecondType::OperatorChebyshev &&
          prec_choice != PrecondType::HypreSmoother)
      {
        exact_element_block_build.reason = "unsupported-preconditioner";
      }
      else if (!is_dg || sfes_.GetVDim() != 1)
      {
        exact_element_block_build.reason = "requires-scalar-dg";
      }
      else if (!spd_ok)
      {
        exact_element_block_build.reason = "requires-spd-operator";
      }
      else if (!exact_face_sum_op)
      {
        exact_element_block_build.reason = "requires-exact-face-operator";
      }
      else if (!direct_identity_system_)
      {
        exact_element_block_build.reason = "requires-direct-identity-system";
      }
      else if (sfes_.GetTrueVSize() != sfes_.GetVSize() ||
               !mfem::IsIdentityProlongation(sfes_.GetProlongationMatrix()))
      {
        exact_element_block_build.reason = "requires-true-local-identity";
      }
      else if (ess_tdof_list_.Size() != 0)
      {
        exact_element_block_build.reason = "requires-no-essential-dofs";
      }
      else
      {
        const mfem::Mesh* mesh = sfes_.GetMesh();
        const bool may_reuse = exact_element_block_step_reuse_requested &&
                               rk_step_index_ >= 0;
        const bool cache_matches =
          may_reuse && exact_element_block_step_cache_ && mesh &&
          exact_element_block_cache_mesh_ == mesh &&
          exact_element_block_cache_mesh_sequence_ == mesh->GetSequence() &&
          exact_element_block_cache_sfes_sequence_ == sfes_.GetSequence() &&
          exact_element_block_cache_rk_step_ == rk_step_index_ &&
          exact_element_block_cache_vector_size_ == A_op->Height() &&
          exact_element_block_cache_alpha_ == alpha_ &&
          exact_element_block_cache_alpha_policy_ ==
            alpha_length_scale_policy_ &&
          exact_element_block_cache_ip_kind_ == ip_kind_ &&
          (exact_face_pair_schwarz_requested ==
           (dynamic_cast<const ExactFacePairAdditiveSchwarz*>(
              exact_element_block_step_cache_.get()) != nullptr));
        if (cache_matches)
        {
          cached_exact_element_block_prec =
            exact_element_block_step_cache_.get();
          using_exact_element_block_jacobi = true;
          exact_element_block_cache_reused = true;
          exact_element_block_build.reason =
            exact_face_pair_schwarz_requested ? "active-face-pair" : "active";
          exact_element_block_build.block_size =
            exact_element_block_cache_block_size_;
          if (const auto* face_pair =
                dynamic_cast<const ExactFacePairAdditiveSchwarz*>(
                  cached_exact_element_block_prec))
          {
            exact_element_block_build.patch_count = face_pair->PatchCount();
            exact_element_block_build.patch_size = face_pair->PatchSize();
            exact_element_block_build.min_overlap = face_pair->MinOverlap();
            exact_element_block_build.max_overlap = face_pair->MaxOverlap();
            exact_element_block_build.restriction_nonzeros =
              face_pair->RestrictionNonzeros();
            exact_element_block_build.inverse_entries =
              face_pair->InverseEntries();
          }
        }
        else
        {
          ::dg_igr::NVTXRange block_range(
            exact_face_pair_schwarz_requested
              ? "IGR/exact-face-pair-schwarz-setup"
              : "IGR/exact-element-block-jacobi-setup");
          exact_element_block_build =
            BuildExactElementBlockJacobi(sfes_,
                                         *exact_face_sum_op,
                                         shared_correlation_block_requested,
                                         exact_face_pair_schwarz_requested);
          if (exact_element_block_build.solver)
          {
            using_exact_element_block_jacobi = true;
            if (may_reuse)
            {
              exact_element_block_step_cache_ =
                std::move(exact_element_block_build.solver);
              exact_element_block_cache_mesh_ = mesh;
              exact_element_block_cache_mesh_sequence_ = mesh->GetSequence();
              exact_element_block_cache_sfes_sequence_ = sfes_.GetSequence();
              exact_element_block_cache_rk_step_ = rk_step_index_;
              exact_element_block_cache_vector_size_ = A_op->Height();
              exact_element_block_cache_block_size_ =
                exact_element_block_build.block_size;
              exact_element_block_cache_alpha_ = alpha_;
              exact_element_block_cache_alpha_policy_ =
                alpha_length_scale_policy_;
              exact_element_block_cache_ip_kind_ = ip_kind_;
              cached_exact_element_block_prec =
                exact_element_block_step_cache_.get();
            }
            else
            {
              exact_element_block_prec =
                std::move(exact_element_block_build.solver);
            }
          }
        }
      }
      exact_element_block_setup_seconds =
        ElapsedSeconds(block_setup_start);
      if (mfem::Mpi::Root())
      {
        std::cout << "[igr-linear] event="
                  << (exact_face_pair_schwarz_requested
                        ? "exact-face-pair-schwarz"
                        : "exact-element-block-jacobi")
                  << " solve_call=" << solve_call
                  << " requested=1"
                  << " effective="
                  << (using_exact_element_block_jacobi ? 1 : 0)
                  << " reason=" << exact_element_block_build.reason
                  << " shared_correlation="
                  << (shared_correlation_block_requested ? 1 : 0)
                  << " cache="
                  << (cached_exact_element_block_prec
                        ? (exact_element_block_cache_reused ? "reused"
                                                            : "built")
                        : "none")
                  << " setup_s=" << exact_element_block_setup_seconds
                  << " colors=" << exact_element_block_build.colors
                  << " block_size=" << exact_element_block_build.block_size
                  << " patch_count=" << exact_element_block_build.patch_count
                  << " patch_size=" << exact_element_block_build.patch_size
                  << " min_overlap=" << exact_element_block_build.min_overlap
                  << " max_overlap=" << exact_element_block_build.max_overlap
                  << " max_faces_per_pair="
                  << exact_element_block_build.max_faces_per_pair
                  << " restriction_nnz="
                  << exact_element_block_build.restriction_nonzeros
                  << " inverse_entries="
                  << exact_element_block_build.inverse_entries
                  << " inverse_bytes="
                  << exact_element_block_build.inverse_entries *
                       static_cast<long long>(sizeof(mfem::real_t))
                  << " overlap_weighting="
                  << (exact_face_pair_schwarz_requested
                        ? "symmetric-inverse-sqrt-multiplicity"
                        : "none")
                  << " pair_scope="
                  << (exact_face_pair_schwarz_requested
                        ? "single-rank-local-interior-faces"
                        : "inactive")
                  << " probe_operator_applies="
                  << exact_element_block_build.operator_applies
                  << " max_symmetry_defect="
                  << exact_element_block_build.max_symmetry_defect
                  << " max_inverse_residual="
                  << exact_element_block_build.max_inverse_residual
                  << std::endl;
      }
    }

    if (original_dg0_two_level_requested &&
        using_exact_element_block_jacobi && exact_element_block_prec)
    {
      const auto coarse_setup_start = Clock::now();
      mfem::Mesh* mesh = sfes_.GetMesh();
      const int ne = sfes_.GetNE();
      const int block_size = exact_element_block_build.block_size;
      MFEM_VERIFY(mesh && ne > 0 && block_size > 1 &&
                    sfes_.GetVSize() == ne * block_size &&
                    sfes_.GetTrueVSize() == sfes_.GetVSize(),
                  "Original-element DG0 screen requires contiguous scalar "
                  "DG element blocks.");

      auto coarse_collection =
        std::make_unique<mfem::DG_FECollection>(0, mesh->Dimension());
      auto coarse_space = std::make_unique<mfem::ParFiniteElementSpace>(
        dynamic_cast<mfem::ParMesh*>(mesh), coarse_collection.get());
      MFEM_VERIFY(coarse_space->GetTrueVSize() == ne,
                  "Original-element DG0 screen requires one local true DOF "
                  "per element.");
      auto coarse_form =
        std::make_unique<mfem::ParBilinearForm>(coarse_space.get());
      coarse_form->AddDomainIntegrator(
        new mfem::MassIntegrator(*invrho_coef_));
      coarse_form->AddInteriorFaceIntegrator(
        new mfem::DGDiffusionIntegrator(*ck_coef_,
                                        -1.0,
                                        IPDGPenalty(sfes_)));
      constexpr int keep_zeros = 0;
      coarse_form->Assemble(keep_zeros);
      coarse_form->Finalize(keep_zeros);
      auto coarse_matrix = std::unique_ptr<mfem::HypreParMatrix>(
        coarse_form->ParallelAssemble());
      MFEM_VERIFY(coarse_matrix && coarse_matrix->Height() == ne &&
                    coarse_matrix->Width() == ne,
                  "Original-element DG0 assembly produced an invalid graph.");
      auto coarse_amg =
        std::make_unique<mfem::HypreBoomerAMG>(*coarse_matrix);
      coarse_amg->SetPrintLevel(0);
      coarse_amg->SetMaxIter(1);
      coarse_amg->SetTol(0.0);

      auto coarse_correction =
        std::make_unique<OriginalElementDG0Correction>(
          sfes_.GetVSize(),
          ne,
          block_size,
          std::move(coarse_collection),
          std::move(coarse_space),
          std::move(coarse_form),
          std::move(coarse_matrix),
          std::move(coarse_amg));
      auto fluctuation_smoother =
        std::make_unique<ElementMeanZeroSolver>(
          std::move(exact_element_block_prec), ne, block_size);
      set_owned_preconditioner(std::make_unique<AdditiveOwnedSolver>(
        std::move(fluctuation_smoother), std::move(coarse_correction)));
      using_original_dg0_two_level = true;
      if (mfem::Mpi::Root())
      {
        std::cout << "[igr-linear] event=original-dg0-two-level"
                  << " solve_call=" << solve_call
                  << " effective=1"
                  << " coarse_dofs=" << ne
                  << " fine_dofs=" << sfes_.GetVSize()
                  << " fluctuation_block_size=" << (block_size - 1)
                  << " setup_s=" << ElapsedSeconds(coarse_setup_start)
                  << " coarse_operator=assembled-current-coefficient-sipg"
                  << " local_operator=exact-mean-zero-element-block"
                  << std::endl;
      }
    }

    if (using_original_dg0_two_level)
    {
      // The orthogonal mean/fluctuation components were installed above.
    }
    else if (using_static_dg_lor_amg)
    {
      if (::dg_igr::EnvFlagEnabledDefault(
            "DG_IGR_IGR_STATIC_DG_LOR_DIAG_SCALE", false))
      {
        auto scaled_lor =
          std::make_unique<SymmetricDiagonalScaledSolver>(
            *static_dg_lor_amg_, static_dg_lor_diagonal_, *A_op);
        if (exact_element_block_prec)
        {
          set_owned_preconditioner(std::make_unique<AdditiveOwnedSolver>(
            std::move(exact_element_block_prec), std::move(scaled_lor)));
        }
        else
        {
          set_owned_preconditioner(std::move(scaled_lor));
        }
      }
      else if (exact_element_block_prec)
      {
        // The unscaled static hierarchy is deliberately not combined: the
        // coefficient-scaled form is the symmetry-preserving co-design being
        // screened here.
        set_owned_preconditioner(std::move(exact_element_block_prec));
        using_static_dg_lor_amg = false;
      }
    }
    else if (using_exact_element_block_jacobi)
    {
      if (cached_exact_element_block_prec)
      {
        prec = cached_exact_element_block_prec;
      }
      else
      {
        set_owned_preconditioner(std::move(exact_element_block_prec));
      }
    }
    else if (prec_choice == PrecondType::BoomerAMG)
    {
      MFEM_VERIFY(A_hypre != nullptr,
                  "BoomerAMG preconditioner requires a Hypre matrix operator.");
      auto amg = std::make_unique<mfem::HypreBoomerAMG>(*A_hypre);
      // Typical AMG-as-preconditioner settings
      amg->SetPrintLevel(0);
      amg->SetMaxIter(1);
      amg->SetTol(0.0);
      if (amg_coarsen_type_ >= 0)
      {
        amg->SetCoarsening(amg_coarsen_type_);
      }
      if (amg_relax_type_ >= 0)
      {
        amg->SetRelaxType(amg_relax_type_);
      }
      if (amg_max_levels_ >= 0)
      {
        amg->SetMaxLevels(amg_max_levels_);
      }
      if (amg_strong_threshold_ >= 0.0)
      {
        amg->SetStrengthThresh(amg_strong_threshold_);
      }
      set_owned_preconditioner(std::move(amg));
    }
    else if (prec_choice == PrecondType::HypreILU)
    {
      MFEM_VERIFY(A_hypre != nullptr,
                  "HypreILU preconditioner requires a Hypre matrix operator.");
      auto ilu = std::make_unique<mfem::HypreILU>();
      ilu->SetPrintLevel(0);
      ilu->SetMaxIter(1);
      ilu->SetTol(0.0);
      ilu->SetOperator(*A_hypre);
      set_owned_preconditioner(std::move(ilu));
    }
    else if (prec_choice == PrecondType::HypreEuclid)
    {
      MFEM_VERIFY(A_hypre != nullptr,
                  "HypreEuclid preconditioner requires a Hypre matrix operator.");
      auto euclid = std::make_unique<mfem::HypreEuclid>(*A_hypre);
      euclid->SetStats(0);
      set_owned_preconditioner(std::move(euclid));
    }
    else if (prec_choice == PrecondType::ParaSails)
    {
      MFEM_VERIFY(A_hypre != nullptr,
                  "ParaSails preconditioner requires a Hypre matrix operator.");
      auto parasails = std::make_unique<mfem::HypreParaSails>(*A_hypre);
      parasails->SetLogging(0);
      set_owned_preconditioner(std::move(parasails));
    }
    else if (prec_choice == PrecondType::HypreSmoother)
    {
      if (A_hypre)
      {
        auto sm = std::make_unique<mfem::HypreSmoother>(*A_hypre);
        switch (smoot_)
        {
        case SmootherType::Jacobi:
          sm->SetType(mfem::HypreSmoother::Jacobi);
          break;
        case SmootherType::L1Jacobi:
          sm->SetType(mfem::HypreSmoother::l1Jacobi);
          break;
        case SmootherType::L1GS:
          sm->SetType(mfem::HypreSmoother::l1GS);
          break;
        case SmootherType::L1GStr:
          sm->SetType(mfem::HypreSmoother::l1GStr);
          break;
        case SmootherType::GS:
          sm->SetType(mfem::HypreSmoother::GS);
          break;
        }
        sm->SetPositiveDiagonal(true);
        set_owned_preconditioner(std::move(sm));
      }
      else
      {
        const bool use_exact_face_diag =
          exact_face_sum_op &&
          ::dg_igr::EnvFlagEnabledDefault(
            "DG_IGR_IGR_EXACT_FACE_DIAGONAL_PRECOND",
            false);
        if (use_exact_face_diag)
        {
          set_owned_preconditioner(
            MakePositiveOperatorJacobi(*A_op, ess_tdof_list_));
          using_operator_jacobi_exact_face_diag = true;
        }
        else
        {
          set_owned_preconditioner(
            MakePositiveOperatorJacobi(*a_, ess_tdof_list_));
        }
        using_operator_jacobi = true;
      }
    }
    else if (prec_choice == PrecondType::OperatorChebyshev)
    {
      ::dg_igr::NVTXRange cheb_range("IGR/chebyshev-setup");
      const bool use_base_operator =
        exact_face_sum_op &&
        ::dg_igr::EnvFlagEnabledDefault(
          "DG_IGR_IGR_CHEB_BASE_OPERATOR",
          false);
      mfem::Operator* cheb_op =
        use_base_operator
          ? &exact_face_sum_op->BaseOperator()
          : A_op;
      mfem::Vector diag(cheb_op->Height());
      const auto cheb_diag_start = Clock::now();
      {
        ::dg_igr::NVTXRange diag_range("IGR/chebyshev-diagonal");
        cheb_op->AssembleDiagonal(diag);
      }
      last_chebyshev_diagonal_seconds_ = ElapsedSeconds(cheb_diag_start);
      total_chebyshev_diagonal_seconds_ +=
        last_chebyshev_diagonal_seconds_;
      const int cheb_order =
        ReadEnvInt("DG_IGR_IGR_CHEB_ORDER",
                   (using_cuda && exact_face_sum_op) ? 3 : 2);
      const int cheb_power_iterations =
        ReadEnvInt("DG_IGR_IGR_CHEB_POWER_ITERATIONS", 10);
      const mfem::real_t cheb_power_tolerance =
        ReadEnvReal("DG_IGR_IGR_CHEB_POWER_TOLERANCE", 1.0e-8);
      const int cheb_power_seed =
        ReadEnvInt("DG_IGR_IGR_CHEB_POWER_SEED", 12345);
      MFEM_VERIFY(cheb_order >= 1 && cheb_order <= 5,
                  "DG_IGR_IGR_CHEB_ORDER must be between 1 and 5 for "
                  "MFEM OperatorChebyshevSmoother.");
      MFEM_VERIFY(cheb_power_iterations > 0,
                  "DG_IGR_IGR_CHEB_POWER_ITERATIONS must be positive.");
      MFEM_VERIFY(cheb_power_tolerance > 0.0,
                  "DG_IGR_IGR_CHEB_POWER_TOLERANCE must be positive.");
      MFEM_VERIFY(cheb_power_seed != 0,
                  "DG_IGR_IGR_CHEB_POWER_SEED must be nonzero.");
      if ((lin_print_ > 0 ||
           ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false)) &&
          mfem::Mpi::Root())
      {
        std::cout << "[igr-linear] event=operator-chebyshev-config"
                  << " order=" << cheb_order
                  << " operator="
                  << (use_base_operator ? "base-pa" : "solve")
                  << " power_iterations=" << cheb_power_iterations
                  << " power_tolerance=" << cheb_power_tolerance
                  << " power_seed=" << cheb_power_seed
                  << std::endl;
      }
      mfem::Operator* cheb_smoother_op = cheb_op;
      if (use_krylov_telemetry)
      {
        cheb_operator_telemetry =
          std::make_unique<KrylovTelemetryOperator>(*cheb_op,
                                                    krylov_telemetry);
        cheb_smoother_op = cheb_operator_telemetry.get();
      }
      const auto cheb_setup_start = Clock::now();
#ifdef MFEM_USE_MPI
      mfem::PowerMethod cheb_power_method(comm);
#else
      mfem::PowerMethod cheb_power_method;
#endif
      mfem::OperatorJacobiSmoother inv_diag(diag, ess_tdof_list_, 1.0);
      mfem::ProductOperator diag_precond(&inv_diag,
                                         cheb_smoother_op,
                                         false,
                                         false);
      mfem::Vector cheb_power_vec(cheb_smoother_op->Width());
      const mfem::real_t max_eig_estimate =
        cheb_power_method.EstimateLargestEigenvalue(diag_precond,
                                                    cheb_power_vec,
                                                    cheb_power_iterations,
                                                    cheb_power_tolerance,
                                                    cheb_power_seed);
      last_chebyshev_eigenvalue_estimate_ =
        static_cast<double>(max_eig_estimate);
      auto cheb = std::make_unique<mfem::OperatorChebyshevSmoother>(
        *cheb_smoother_op,
        diag,
        ess_tdof_list_,
        cheb_order,
        max_eig_estimate);
      last_chebyshev_setup_seconds_ = ElapsedSeconds(cheb_setup_start);
      total_chebyshev_setup_seconds_ += last_chebyshev_setup_seconds_;
      ++total_chebyshev_smoother_builds_;
      set_owned_preconditioner(std::move(cheb));
      using_operator_chebyshev = true;
      using_operator_chebyshev_base = use_base_operator;
    }
    // else None -> leave prec null
  }
  const auto pc_setup_stop = std::chrono::steady_clock::now();
  last_pc_setup_seconds_ =
    std::chrono::duration<double>(pc_setup_stop - pc_setup_start).count();
  total_pc_setup_seconds_ += last_pc_setup_seconds_;
  if (run_operator_probe && prec &&
      ::dg_igr::EnvFlagEnabledDefault(
        "DG_IGR_IGR_PRECONDITIONER_PROBE", false))
  {
    LogOperatorAlgebraProbe(
      "preconditioner", solve_call, *prec, comm, operator_probe_steps);
  }
  krylov_telemetry.ResetCounters();

  if (lin_print_ > 0)
  {
    LogIGRSelection(warm_start,
                    is_dg,
                    spd_ok,
                    using_cuda,
                    solver_choice,
                    prec_choice,
                    lin_rtol_,
                    lin_atol_,
                    lin_maxit_);
  }

  // Warm start is only safe if the prior solve produced a usable iterate.
  // If not, fall back to a cold-start solve to avoid poisoning Krylov with
  // stale/non-finite data.
  bool enable_warm_start = warm_start;
  if (enable_warm_start && GetProductionLinearSolveCalls() > 0)
  {
    if (!prev_linear_converged || !std::isfinite(prev_linear_final_norm))
    {
      enable_warm_start = false;
      if (lin_print_ > 0)
      {
        LogIGRFallback("warm-start-disabled-after-unsuccessful-previous-solve",
                       solver_choice,
                       solver_choice,
                       prec_choice,
                       prec_choice);
      }
    }
  }
  if (!enable_warm_start || !VectorIsFinite(X_))
  {
    X_ = 0.0;
    enable_warm_start = false;
  }
  MFEM_VERIFY(VectorIsFinite(B_),
              "IGR RHS contains non-finite values before Krylov solve.");

  last_warm_start_guard_rejected_ = false;
  last_warm_start_residual_norm_ =
    std::numeric_limits<double>::quiet_NaN();
  last_zero_start_residual_norm_ =
    std::numeric_limits<double>::quiet_NaN();
  if (enable_warm_start &&
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_WARM_START_GUARD", false))
  {
    ::dg_igr::NVTXRange warm_guard_range("IGR/warm-start-residual-guard");
    mfem::Vector operator_guess(B_.Size());
    mfem::Vector warm_residual(B_);
    A_op->Mult(X_, operator_guess);
    warm_residual -= operator_guess;

    auto residual_norm = [&](const mfem::Vector& residual)
    {
      if (!prec)
      {
        return OperatorProbeNorm(residual, comm);
      }
      mfem::Vector preconditioned(residual.Size());
      prec->Mult(residual, preconditioned);
      const double norm_squared =
        OperatorProbeDot(residual, preconditioned, comm);
      return norm_squared >= 0.0
               ? std::sqrt(norm_squared)
               : std::numeric_limits<double>::quiet_NaN();
    };

    last_warm_start_residual_norm_ = residual_norm(warm_residual);
    last_zero_start_residual_norm_ = residual_norm(B_);
    const bool warm_residual_usable =
      std::isfinite(last_warm_start_residual_norm_) &&
      std::isfinite(last_zero_start_residual_norm_) &&
      last_warm_start_residual_norm_ <= last_zero_start_residual_norm_;
    if (!warm_residual_usable)
    {
      X_ = 0.0;
      enable_warm_start = false;
      last_warm_start_guard_rejected_ = true;
      if (mfem::Mpi::Root())
      {
        std::cout << "[igr-linear] event=warm-start-residual-guard"
                  << " solve_call=" << solve_call
                  << " rk_step=" << rk_step_index_
                  << " rk_stage=" << rk_stage_index_
                  << " action=cold-start"
                  << " warm_residual_norm="
                  << last_warm_start_residual_norm_
                  << " zero_residual_norm="
                  << last_zero_start_residual_norm_
                  << std::endl;
      }
    }
  }
  last_warm_start_effective_ = enable_warm_start;

  const bool is_intermediate_stage =
    mfem::IsIntermediateRKStage(rk_stage_index_, rk_stage_count_);
  const double effective_rtol =
    is_intermediate_stage ? (lin_rtol_ * stage_rtol_scale_) : lin_rtol_;
  const bool reduction_free_chebyshev_requested =
    ::dg_igr::EnvFlagEnabledDefault(
      "DG_IGR_IGR_REDUCTION_FREE_CHEBYSHEV", false);
  const int reduction_free_chebyshev_degree =
    ReadEnvInt("DG_IGR_IGR_REDUCTION_FREE_CHEBYSHEV_DEGREE", 64);
  const char* reduction_free_lower_text =
    std::getenv("DG_IGR_IGR_REDUCTION_FREE_CHEBYSHEV_LAMBDA_MIN");
  const char* reduction_free_upper_text =
    std::getenv("DG_IGR_IGR_REDUCTION_FREE_CHEBYSHEV_LAMBDA_MAX");
  const bool reduction_free_bounds_supplied =
    reduction_free_lower_text && reduction_free_lower_text[0] != '\0' &&
    reduction_free_upper_text && reduction_free_upper_text[0] != '\0';
  const double reduction_free_bound_safety =
    ReadEnvReal("DG_IGR_IGR_REDUCTION_FREE_CHEBYSHEV_BOUND_SAFETY", 1.05);
  const double reduction_free_input_lambda_min =
    reduction_free_bounds_supplied
      ? ReadEnvReal("DG_IGR_IGR_REDUCTION_FREE_CHEBYSHEV_LAMBDA_MIN", 0.0)
      : std::numeric_limits<double>::quiet_NaN();
  const double reduction_free_input_lambda_max =
    reduction_free_bounds_supplied
      ? ReadEnvReal("DG_IGR_IGR_REDUCTION_FREE_CHEBYSHEV_LAMBDA_MAX", 0.0)
      : std::numeric_limits<double>::quiet_NaN();
  if (reduction_free_chebyshev_requested)
  {
    MFEM_VERIFY(reduction_free_chebyshev_degree >= 1 &&
                  reduction_free_chebyshev_degree <= 4096,
                "DG_IGR_IGR_REDUCTION_FREE_CHEBYSHEV_DEGREE must be "
                "between 1 and 4096.");
    MFEM_VERIFY(reduction_free_bounds_supplied,
                "DG_IGR_IGR_REDUCTION_FREE_CHEBYSHEV requires explicit "
                "LAMBDA_MIN and LAMBDA_MAX bounds for the preconditioned "
                "operator.");
    MFEM_VERIFY(std::isfinite(reduction_free_input_lambda_min) &&
                  std::isfinite(reduction_free_input_lambda_max) &&
                  reduction_free_input_lambda_min > 0.0 &&
                  reduction_free_input_lambda_max >
                    reduction_free_input_lambda_min,
                "Reduction-free Chebyshev spectral bounds must satisfy "
                "0 < LAMBDA_MIN < LAMBDA_MAX.");
    MFEM_VERIFY(std::isfinite(reduction_free_bound_safety) &&
                  reduction_free_bound_safety >= 1.0,
                "DG_IGR_IGR_REDUCTION_FREE_CHEBYSHEV_BOUND_SAFETY must "
                "be finite and at least one.");
  }
  const double reduction_free_lambda_min =
    reduction_free_input_lambda_min / reduction_free_bound_safety;
  const double reduction_free_lambda_max =
    reduction_free_input_lambda_max * reduction_free_bound_safety;
  if (reduction_free_chebyshev_requested)
  {
    MFEM_VERIFY(std::isfinite(reduction_free_lambda_min) &&
                  std::isfinite(reduction_free_lambda_max) &&
                  reduction_free_lambda_min > 0.0 &&
                  reduction_free_lambda_max > reduction_free_lambda_min,
                "Reduction-free Chebyshev safety expansion produced an "
                "invalid spectral interval.");
  }
  std::string reduction_free_chebyshev_reason = "not-requested";
  bool reduction_free_chebyshev_effective = false;
  if (reduction_free_chebyshev_requested)
  {
    if (solver_choice != LinearSolverType::CG)
    {
      reduction_free_chebyshev_reason = "requires-cg-fallback";
    }
    else if (!spd_ok)
    {
      reduction_free_chebyshev_reason = "requires-spd-operator";
    }
    else if (!prec)
    {
      reduction_free_chebyshev_reason = "requires-spd-preconditioner";
    }
    else if (sizeof(mfem::real_t) != sizeof(double))
    {
      reduction_free_chebyshev_reason = "requires-fp64-mfem";
    }
    else
    {
      reduction_free_chebyshev_effective = true;
      reduction_free_chebyshev_reason = "active";
    }
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=reduction-free-chebyshev-config"
                << " solve_call=" << solve_call
                << " requested=1"
                << " effective="
                << (reduction_free_chebyshev_effective ? 1 : 0)
                << " reason=" << reduction_free_chebyshev_reason
                << " degree=" << reduction_free_chebyshev_degree
                << " input_lambda_min="
                << reduction_free_input_lambda_min
                << " input_lambda_max="
                << reduction_free_input_lambda_max
                << " bound_safety=" << reduction_free_bound_safety
                << " effective_lambda_min=" << reduction_free_lambda_min
                << " effective_lambda_max=" << reduction_free_lambda_max
                << " fp64="
                << (sizeof(mfem::real_t) == sizeof(double) ? 1 : 0)
                << std::endl;
    }
  }
  mfem::Vector event_initial_guess;
  if (event_capture_enabled || true_initial_residual_requested)
  {
    event_initial_guess = X_;
  }

  std::unique_ptr<KrylovTelemetryOperator> solve_operator_telemetry;
  std::unique_ptr<KrylovTelemetryPreconditioner> telemetry_preconditioner;
  mfem::Operator* solve_operator = A_op;
  mfem::Solver* solve_preconditioner = prec;
  if (use_krylov_telemetry)
  {
    solve_operator_telemetry =
      std::make_unique<KrylovTelemetryOperator>(
        *A_op,
        krylov_telemetry,
        exact_face_sum_op ? &exact_face_sum_op->BaseOperator() : nullptr,
        exact_face_sum_op ? &exact_face_sum_op->FaceOperator() : nullptr,
        diagnostic_state_,
        &sfes_,
        &negative_quadratic_capture_written_,
        comm,
        solve_call);
    solve_operator = solve_operator_telemetry.get();
    if (prec)
    {
      telemetry_preconditioner =
        std::make_unique<KrylovTelemetryPreconditioner>(*prec,
                                                        krylov_telemetry);
      solve_preconditioner = telemetry_preconditioner.get();
    }
  }

  if (true_initial_residual_requested)
  {
    mfem::Vector operator_initial(B_.Size());
    mfem::Vector initial_residual(B_);
    A_op->Mult(X_, operator_initial);
    initial_residual -= operator_initial;
    last_linear_true_initial_residual_abs_ =
      OperatorProbeNorm(initial_residual, comm);
    if (prec)
    {
      mfem::Vector preconditioned_initial(B_.Size());
      prec->Mult(initial_residual, preconditioned_initial);
      const double norm_squared =
        OperatorProbeDot(initial_residual, preconditioned_initial, comm);
      last_linear_recomputed_preconditioned_initial_norm_ =
        norm_squared >= 0.0 ? std::sqrt(norm_squared)
                            : std::numeric_limits<double>::quiet_NaN();
    }
    else
    {
      last_linear_recomputed_preconditioned_initial_norm_ =
        last_linear_true_initial_residual_abs_;
    }
  }

  auto configure_solver = [&](mfem::IterativeSolver& solver,
                              double relative_tolerance,
                              bool iterative_mode)
  {
    solver.SetOperator(*solve_operator);
    if (solve_preconditioner)
    {
      solver.SetPreconditioner(*solve_preconditioner);
    }
    solver.SetRelTol(relative_tolerance);
    solver.SetAbsTol(lin_atol_);
    solver.SetMaxIter(lin_maxit_);
    solver.SetPrintLevel(lin_print_);
    solver.iterative_mode = iterative_mode;
  };

  if (exact_face_sum_op)
  {
    exact_face_sum_op->ResetTiming();
  }

  const auto solve_start = std::chrono::steady_clock::now();

  auto run_krylov = [&](double relative_tolerance, bool iterative_mode)
  {
    ::dg_igr::NVTXRange krylov_range("IGR/krylov-solve");
    if (solver_choice == LinearSolverType::CG)
    {
      mfem::CGSolver solver(comm);
      configure_solver(solver, relative_tolerance, iterative_mode);
      solver.Mult(B_, X_);
      last_linear_iters_ = solver.GetNumIterations();
      last_linear_converged_ = solver.GetConverged();
      last_linear_final_norm_ = solver.GetFinalNorm();
      last_linear_initial_norm_ = solver.GetInitialNorm();
    }
    else if (solver_choice == LinearSolverType::GMRES)
    {
      mfem::GMRESSolver solver(comm);
      configure_solver(solver, relative_tolerance, iterative_mode);
      solver.Mult(B_, X_);
      last_linear_iters_ = solver.GetNumIterations();
      last_linear_converged_ = solver.GetConverged();
      last_linear_final_norm_ = solver.GetFinalNorm();
      last_linear_initial_norm_ = solver.GetInitialNorm();
    }
    else if (solver_choice == LinearSolverType::FGMRES)
    {
      mfem::FGMRESSolver solver(comm);
      configure_solver(solver, relative_tolerance, iterative_mode);
      solver.Mult(B_, X_);
      last_linear_iters_ = solver.GetNumIterations();
      last_linear_converged_ = solver.GetConverged();
      last_linear_final_norm_ = solver.GetFinalNorm();
      last_linear_initial_norm_ = solver.GetInitialNorm();
    }
    else if (solver_choice == LinearSolverType::BiCGSTAB)
    {
      mfem::BiCGSTABSolver solver(comm);
      configure_solver(solver, relative_tolerance, iterative_mode);
      solver.Mult(B_, X_);
      last_linear_iters_ = solver.GetNumIterations();
      last_linear_converged_ = solver.GetConverged();
      last_linear_final_norm_ = solver.GetFinalNorm();
      last_linear_initial_norm_ = solver.GetInitialNorm();
    }
    else if (solver_choice == LinearSolverType::MINRES)
    {
      mfem::MINRESSolver solver(comm);
      configure_solver(solver, relative_tolerance, iterative_mode);
      solver.Mult(B_, X_);
      last_linear_iters_ = solver.GetNumIterations();
      last_linear_converged_ = solver.GetConverged();
      last_linear_final_norm_ = solver.GetFinalNorm();
      last_linear_initial_norm_ = solver.GetInitialNorm();
    }
    else
    {
      MFEM_ABORT("Unsupported IGR linear solver selection.");
    }
  };

  bool reduction_free_chebyshev_attempted = false;
  bool reduction_free_chebyshev_accepted = false;
  bool reduction_free_chebyshev_fallback = false;
  bool reduction_free_chebyshev_nonfinite = false;
  bool reduction_free_chebyshev_restored_original = false;
  double reduction_free_chebyshev_seconds = 0.0;
  double reduction_free_certificate_seconds = 0.0;
  double reduction_free_operator_local_reduction_seconds = 0.0;
  double reduction_free_collective_seconds = 0.0;
  double reduction_free_fallback_seconds = 0.0;
  double reduction_free_candidate_residual_abs =
    std::numeric_limits<double>::quiet_NaN();
  double reduction_free_candidate_residual_rel_rhs =
    std::numeric_limits<double>::quiet_NaN();

  if (reduction_free_chebyshev_effective)
  {
    ::dg_igr::NVTXRange polynomial_range(
      "IGR/reduction-free-preconditioned-chebyshev");
    reduction_free_chebyshev_attempted = true;
    // Keep X_ transactional.  A bad polynomial interval must not contaminate
    // either the original warm start or the bounded Krylov fallback.
    mfem::Vector candidate(X_);
    mfem::Vector residual(B_.Size());
    mfem::Vector operator_iterate(B_.Size());
    mfem::Vector preconditioned_residual(B_.Size());
    mfem::Vector direction(B_.Size());

    // The interval is supplied explicitly and expanded by a safety factor.
    // For SPD A and SPD P, the left-preconditioned operator P A is similar to
    // the SPD operator P^(1/2) A P^(1/2), so the standard Chebyshev recurrence
    // applies.  Its fixed-degree loop contains operator/preconditioner applies
    // and vector updates only: no dot products or global reductions.
    // These half-sums avoid overflowing an otherwise valid finite interval.
    const double theta = 0.5 * reduction_free_lambda_max +
                         0.5 * reduction_free_lambda_min;
    const double delta = 0.5 * reduction_free_lambda_max -
                         0.5 * reduction_free_lambda_min;
    const double spectral_sigma = theta / delta;
    MFEM_VERIFY(std::isfinite(theta) && std::isfinite(delta) &&
                  std::isfinite(spectral_sigma) && delta > 0.0 &&
                  spectral_sigma > 1.0,
                "Reduction-free Chebyshev received an unusable spectral "
                "interval.");
    double rho_previous = 1.0 / spectral_sigma;

    const auto polynomial_start = Clock::now();
    solve_operator->Mult(candidate, operator_iterate);
    residual = B_;
    residual -= operator_iterate;
    solve_preconditioner->Mult(residual, preconditioned_residual);
    direction = preconditioned_residual;
    direction *= 1.0 / theta;
    candidate += direction;

    for (int iteration = 1;
         iteration < reduction_free_chebyshev_degree;
         ++iteration)
    {
      solve_operator->Mult(candidate, operator_iterate);
      residual = B_;
      residual -= operator_iterate;
      solve_preconditioner->Mult(residual, preconditioned_residual);
      const double rho = 1.0 / (2.0 * spectral_sigma - rho_previous);
      direction *= rho * rho_previous;
      direction.Add((2.0 * rho) / delta, preconditioned_residual);
      candidate += direction;
      rho_previous = rho;
    }
    MFEM_DEVICE_SYNC;
    reduction_free_chebyshev_seconds = ElapsedSeconds(polynomial_start);

    const auto certificate_start = Clock::now();
    // This bypasses telemetry and applies the authoritative FP64 SIPG
    // operator independently.  Use MFEM's established FP64 vector reductions
    // for the two local squared norms.  The experimental custom pair reducer
    // was not safe on the SM120 CUDA memory path; a single two-scalar MPI
    // collective still certifies the result without a per-iteration global
    // reduction.
    const auto operator_local_start = Clock::now();
    A_op->Mult(candidate, operator_iterate);
    residual = B_;
    residual -= operator_iterate;
    const double local_norm_squares[2] = {
      static_cast<double>(residual * residual),
      static_cast<double>(B_ * B_)};
    reduction_free_operator_local_reduction_seconds =
      ElapsedSeconds(operator_local_start);

    double global_norm_squares[2] = {0.0, 0.0};
    const auto collective_start = Clock::now();
    MPI_Allreduce(local_norm_squares,
                  global_norm_squares,
                  2,
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);
    reduction_free_collective_seconds = ElapsedSeconds(collective_start);

    const bool squared_norms_finite =
      std::isfinite(global_norm_squares[0]) &&
      std::isfinite(global_norm_squares[1]) &&
      global_norm_squares[0] >= 0.0 && global_norm_squares[1] >= 0.0;
    if (squared_norms_finite)
    {
      reduction_free_candidate_residual_abs =
        std::sqrt(global_norm_squares[0]);
      const double rhs_norm = std::sqrt(global_norm_squares[1]);
      reduction_free_candidate_residual_rel_rhs =
        reduction_free_candidate_residual_abs /
        std::max(rhs_norm, 1.0e-300);
    }
    reduction_free_chebyshev_nonfinite =
      !squared_norms_finite ||
      !std::isfinite(reduction_free_candidate_residual_abs) ||
      !std::isfinite(reduction_free_candidate_residual_rel_rhs);
    const bool absolute_pass =
      !reduction_free_chebyshev_nonfinite &&
      reduction_free_candidate_residual_abs <= lin_atol_;
    const bool relative_pass =
      !reduction_free_chebyshev_nonfinite &&
      reduction_free_candidate_residual_rel_rhs <= effective_rtol;
    reduction_free_chebyshev_accepted = absolute_pass || relative_pass;
    reduction_free_certificate_seconds = ElapsedSeconds(certificate_start);

    last_linear_iters_ = reduction_free_chebyshev_degree;
    last_linear_final_norm_ = reduction_free_candidate_residual_abs;
    last_linear_true_residual_abs_ =
      reduction_free_candidate_residual_abs;
    last_linear_true_residual_rel_rhs_ =
      reduction_free_candidate_residual_rel_rhs;
    if (std::isfinite(last_linear_true_initial_residual_abs_))
    {
      last_linear_true_residual_rel_initial_ =
        reduction_free_candidate_residual_abs /
        std::max(last_linear_true_initial_residual_abs_, 1.0e-300);
    }
    last_linear_converged_ = reduction_free_chebyshev_accepted;

    if (reduction_free_chebyshev_accepted)
    {
      X_ = candidate;
    }
    else
    {
      reduction_free_chebyshev_fallback = true;
      // X_ was never modified, so fallback starts from the exact original
      // guess even when the candidate became non-finite.
      reduction_free_chebyshev_restored_original = true;
      const int polynomial_iterations = last_linear_iters_;
      // A failed research candidate is never allowed to poison Krylov: CG
      // always restarts from the exact original guess.
      const auto fallback_start = Clock::now();
      run_krylov(effective_rtol, /*iterative_mode=*/true);
      MFEM_DEVICE_SYNC;
      reduction_free_fallback_seconds = ElapsedSeconds(fallback_start);
      last_linear_iters_ += polynomial_iterations;
    }

    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=reduction-free-chebyshev-result"
                << " solve_call=" << solve_call
                << " rk_step=" << rk_step_index_
                << " rk_stage=" << rk_stage_index_
                << " degree=" << reduction_free_chebyshev_degree
                << " lambda_min=" << reduction_free_lambda_min
                << " lambda_max=" << reduction_free_lambda_max
                << " initial_residual_reduction=elided"
                << " candidate_residual_abs="
                << reduction_free_candidate_residual_abs
                << " candidate_residual_rel_rhs="
                << reduction_free_candidate_residual_rel_rhs
                << " rtol=" << effective_rtol
                << " atol=" << lin_atol_
                << " accepted="
                << (reduction_free_chebyshev_accepted ? 1 : 0)
                << " fallback="
                << (reduction_free_chebyshev_fallback ? 1 : 0)
                << " candidate_nonfinite="
                << (reduction_free_chebyshev_nonfinite ? 1 : 0)
                << " restored_original_guess="
                << (reduction_free_chebyshev_restored_original ? 1 : 0)
                << " restore_reason="
                << (reduction_free_chebyshev_restored_original
                      ? "certificate-failed"
                      : "none")
                << " polynomial_s=" << reduction_free_chebyshev_seconds
                << " certificate_s=" << reduction_free_certificate_seconds
                << " operator_local_reduction_s="
                << reduction_free_operator_local_reduction_seconds
                << " collective_s=" << reduction_free_collective_seconds
                << " local_norm_reductions=2"
                << " mpi_norm_collectives=1"
                << " reduced_fp64_scalars=2"
                << " fallback_s=" << reduction_free_fallback_seconds
                << std::endl;
    }
  }
  else
  {
    run_krylov(effective_rtol, enable_warm_start);
  }

  const bool certify_extended_solve =
    extended_certificate_enabled &&
    last_linear_iters_ >= extended_certificate_start;
  const bool certify_independent_residual =
    certify_current_solve_ || certify_extended_solve ||
    reduction_free_chebyshev_attempted;
  auto recompute_true_residual = [&]()
  {
    mfem::Vector residual(B_);
    mfem::Vector operator_solution(B_.Size());
    A_op->Mult(X_, operator_solution);
    residual -= operator_solution;
    last_linear_true_residual_abs_ = OperatorProbeNorm(residual, comm);
    const double rhs_norm = OperatorProbeNorm(B_, comm);
    last_linear_true_residual_rel_rhs_ =
      last_linear_true_residual_abs_ / std::max(rhs_norm, 1.0e-300);
    last_linear_true_residual_rel_initial_ =
      last_linear_true_residual_abs_ /
      std::max(last_linear_true_initial_residual_abs_, 1.0e-300);
    if (true_residual_requested)
    {
      if (prec)
      {
        mfem::Vector preconditioned_residual(B_.Size());
        prec->Mult(residual, preconditioned_residual);
        const double norm_squared =
          OperatorProbeDot(residual, preconditioned_residual, comm);
        last_linear_recomputed_preconditioned_final_norm_ =
          norm_squared >= 0.0 ? std::sqrt(norm_squared)
                              : std::numeric_limits<double>::quiet_NaN();
      }
      else
      {
        last_linear_recomputed_preconditioned_final_norm_ =
          last_linear_true_residual_abs_;
      }
    }
  };
  const bool reduction_free_certificate_is_current =
    reduction_free_chebyshev_accepted &&
    !reduction_free_chebyshev_fallback;
  if ((true_residual_requested || certify_independent_residual) &&
      !reduction_free_certificate_is_current)
  {
    recompute_true_residual();
  }

  bool independent_certificate_passed = true;
  int residual_replacement_retries = 0;
  if (certify_independent_residual)
  {
    const double certificate_factor =
      reduction_free_chebyshev_attempted ? 1.0
                                         : extended_certificate_factor;
    const double certificate_relative_limit =
      certificate_factor * effective_rtol;
    const double certificate_absolute_limit =
      certificate_factor * lin_atol_;
    last_independent_certificate_relative_limit_ = certificate_relative_limit;
    last_independent_certificate_absolute_limit_ = certificate_absolute_limit;
    auto certificate_passes = [&]()
    {
      const bool relative_pass =
        std::isfinite(last_linear_true_residual_rel_rhs_) &&
        last_linear_true_residual_rel_rhs_ <= certificate_relative_limit;
      const bool absolute_pass =
        std::isfinite(last_linear_true_residual_abs_) &&
        last_linear_true_residual_abs_ <= certificate_absolute_limit;
      return relative_pass || absolute_pass;
    };
    independent_certificate_passed = certificate_passes();
    int local_solution_finite = VectorIsFinite(X_) ? 1 : 0;
    int collective_solution_finite = 0;
    MPI_Allreduce(&local_solution_finite,
                  &collective_solution_finite,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  comm);
    if (!independent_certificate_passed && collective_solution_finite != 0)
    {
      const int first_attempt_iters = last_linear_iters_;
      const double first_attempt_initial_norm = last_linear_initial_norm_;
      const double first_attempt_true_rel = last_linear_true_residual_rel_rhs_;
      if (mfem::Mpi::Root())
      {
        std::cout << "[igr-linear] event=residual-replacement-retry"
                  << " solve_call=" << solve_call
                  << " rk_step=" << rk_step_index_
                  << " rk_stage=" << rk_stage_index_
                  << " attempt=2"
                  << " first_attempt_iters=" << first_attempt_iters
                  << " first_true_residual_rel_rhs="
                  << first_attempt_true_rel
                  << " relative_limit=" << certificate_relative_limit
                  << " absolute_limit=" << certificate_absolute_limit
                  << " action=restart-recurrence-from-current-iterate"
                  << std::endl;
      }

      run_krylov(effective_rtol, /*iterative_mode=*/true);
      last_linear_iters_ += first_attempt_iters;
      last_linear_initial_norm_ = first_attempt_initial_norm;
      ++residual_replacement_retries;
      recompute_true_residual();
      independent_certificate_passed = certificate_passes();
    }
    // The independent residual is authoritative for certified solves. This
    // also recognizes a valid capped iterate when the Krylov recurrence's
    // reported norm has drifted from the true residual.
    last_linear_converged_ = independent_certificate_passed;
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=independent-residual-certificate"
                << " solve_call=" << solve_call
                << " rk_step=" << rk_step_index_
                << " rk_stage=" << rk_stage_index_
                << " iterations=" << last_linear_iters_
                << " reason="
                << (reduction_free_chebyshev_attempted
                      ? "reduction-free-chebyshev"
                      : (certify_current_solve_ ? "restart"
                                                : "extended-solve"))
                << " start=" << extended_certificate_start
                << " factor=" << certificate_factor
                << " relative_limit=" << certificate_relative_limit
                << " absolute_limit=" << certificate_absolute_limit
                << " true_residual_rel_rhs="
                << last_linear_true_residual_rel_rhs_
                << " true_residual_rel_initial="
                << last_linear_true_residual_rel_initial_
                << " residual_replacement_retries="
                << residual_replacement_retries
                << " passed=" << (independent_certificate_passed ? 1 : 0)
                << std::endl;
    }
  }

  const auto solve_stop = std::chrono::steady_clock::now();
  last_linear_solve_seconds_ =
    std::chrono::duration<double>(solve_stop - solve_start).count();
  total_linear_solve_seconds_ += last_linear_solve_seconds_;
  last_krylov_operator_apply_seconds_ =
    krylov_telemetry.operator_apply_seconds;
  last_krylov_preconditioner_apply_seconds_ =
    krylov_telemetry.preconditioner_apply_seconds;
  last_krylov_preconditioner_operator_apply_seconds_ =
    krylov_telemetry.preconditioner_operator_apply_seconds;
  last_krylov_operator_apply_count_ =
    krylov_telemetry.operator_apply_count;
  last_krylov_preconditioner_apply_count_ =
    krylov_telemetry.preconditioner_apply_count;
  last_krylov_preconditioner_operator_apply_count_ =
    krylov_telemetry.preconditioner_operator_apply_count;
  total_krylov_operator_apply_seconds_ +=
    last_krylov_operator_apply_seconds_;
  total_krylov_preconditioner_apply_seconds_ +=
    last_krylov_preconditioner_apply_seconds_;
  total_krylov_preconditioner_operator_apply_seconds_ +=
    last_krylov_preconditioner_operator_apply_seconds_;
  total_krylov_operator_apply_count_ +=
    last_krylov_operator_apply_count_;
  total_krylov_preconditioner_apply_count_ +=
    last_krylov_preconditioner_apply_count_;
  total_krylov_preconditioner_operator_apply_count_ +=
    last_krylov_preconditioner_operator_apply_count_;
  ++linear_solve_calls_;
  total_linear_iters_ += last_linear_iters_;
  max_linear_iters_ = std::max(max_linear_iters_, last_linear_iters_);
  if (last_linear_converged_)
  {
    ++total_linear_converged_calls_;
  }

  last_linear_solver_name_ = LinearSolverTypeName(solver_choice);
  last_precond_name_ = using_static_dg_lor_amg
                         ? "static-dg-lor-amg"
                         : (using_exact_element_block_jacobi
                              ? (exact_face_pair_schwarz_requested
                                   ? "exact-face-pair-schwarz"
                                   : (shared_correlation_block_requested
                                        ? "shared-correlation-block-jacobi"
                                        : "exact-element-block-jacobi"))
                         : (using_operator_chebyshev
                              ? (using_operator_chebyshev_base
                                   ? "operatorchebyshev-base"
                                   : "operatorchebyshev")
                              : (using_operator_jacobi_exact_face_diag
                                   ? "operatorjacobi-exactface"
                                   : (using_operator_jacobi
                                        ? "operatorjacobi"
                                        : PrecondTypeName(prec_choice)))));

  const bool capture_linear_event =
    event_capture_enabled && !linear_event_capture_written_ &&
    (!last_linear_converged_ || last_linear_iters_ >= event_capture_min_iters);
  if (capture_linear_event)
  {
    SaveLinearEventCapture(event_capture_prefix_text,
                           solve_call,
                           rk_step_index_,
                           rk_stage_index_,
                           current_time_,
                           current_dt_,
                           *diagnostic_state_,
                           sfes_,
                           B_,
                           event_initial_guess,
                           X_,
                           last_linear_iters_,
                           last_linear_converged_,
                           last_linear_initial_norm_,
                           last_linear_final_norm_,
                           last_linear_true_initial_residual_abs_,
                           last_linear_true_residual_abs_,
                           last_linear_true_residual_rel_rhs_,
                           last_linear_true_residual_rel_initial_,
                           last_linear_recomputed_preconditioned_initial_norm_,
                           last_linear_recomputed_preconditioned_final_norm_,
                           effective_rtol,
                           warm_start,
                           enable_warm_start,
                           last_linear_solver_name_,
                           last_precond_name_,
                           last_chebyshev_eigenvalue_estimate_);
    linear_event_capture_written_ = true;
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=solve-capture"
                << " solve_call=" << solve_call
                << " capture_dir=" << event_capture_prefix_text << "/solve-"
                << solve_call << std::endl;
    }
  }

  const bool log_linear_event =
    !last_linear_converged_ ||
    (event_log_min_iters > 0 && last_linear_iters_ >= event_log_min_iters);
  if (log_linear_event && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=solve"
              << " solve_call=" << solve_call
              << " purpose="
              << (current_solve_purpose_ == IGRSolvePurpose::OutputOnly
                    ? "output-only"
                    : (current_solve_purpose_ ==
                           IGRSolvePurpose::ProductionStage0Prefetch
                         ? "stage0-prefetch"
                         : "stage"))
              << " rk_step=" << rk_step_index_
              << " rk_stage=" << rk_stage_index_
              << " time=" << current_time_
              << " dt=" << current_dt_
              << " iters=" << last_linear_iters_
              << " converged=" << (last_linear_converged_ ? 1 : 0)
              << " reported_initial_norm=" << last_linear_initial_norm_
              << " reported_final_norm=" << last_linear_final_norm_
              << " true_initial_residual_abs="
              << last_linear_true_initial_residual_abs_
              << " true_residual_abs=" << last_linear_true_residual_abs_
              << " true_residual_rel_rhs="
              << last_linear_true_residual_rel_rhs_
              << " true_residual_rel_initial="
              << last_linear_true_residual_rel_initial_
              << " recomputed_preconditioned_initial_norm="
              << last_linear_recomputed_preconditioned_initial_norm_
              << " recomputed_preconditioned_final_norm="
              << last_linear_recomputed_preconditioned_final_norm_
              << " effective_rtol=" << effective_rtol
              << " warm_requested=" << (warm_start ? 1 : 0)
              << " warm_effective=" << (enable_warm_start ? 1 : 0)
              << " solver=" << last_linear_solver_name_
              << " preconditioner=" << last_precond_name_
              << " cheb_eigen=" << last_chebyshev_eigenvalue_estimate_
              << std::endl;
  }

  if (!last_linear_converged_)
  {
    if (nonconvergence_policy_ == NonconvergencePolicy::Accept)
    {
      MFEM_VERIFY(VectorIsFinite(X_),
                  "IGR linear solve produced a non-finite capped iterate; "
                  "nonconvergence policy 'accept' cannot recover it into "
                  "sigma.");
      if (mfem::Mpi::Root())
      {
        std::cout << "[igr-linear] event=nonconvergence-policy"
                  << " policy=accept"
                  << " solve_call=" << solve_call
                  << " rk_step=" << rk_step_index_
                  << " rk_stage=" << rk_stage_index_
                  << " time=" << current_time_
                  << " iters=" << last_linear_iters_
                  << " action=recover-max-iteration-iterate"
                  << std::endl;
      }
    }
    else
    {
      MFEM_ABORT("IGR linear solve did not converge: solve_call="
                 << solve_call << " rk_step=" << rk_step_index_
                 << " rk_stage=" << rk_stage_index_ << " time="
                 << current_time_ << " iterations=" << last_linear_iters_
                 << " solver=" << last_linear_solver_name_
                 << " preconditioner=" << last_precond_name_
                 << ". The failed iterate was not recovered into sigma. "
                    "Use --igr-nonconvergence-policy accept to recover the "
                    "finite capped iterate and continue.");
    }
  }

  if (exact_face_sum_op)
  {
    ++exact_face_operator_uses_;
    long long local_mult_calls = exact_face_sum_op->LastMultCalls();
    long long max_mult_calls = local_mult_calls;
#ifdef MFEM_USE_MPI
    MPI_Allreduce(&local_mult_calls,
                  &max_mult_calls,
                  1,
                  MPI_LONG_LONG,
                  MPI_MAX,
                  comm);
#endif
    last_exact_face_operator_mult_calls_ = max_mult_calls;
  }
  else
  {
    last_exact_face_operator_mult_calls_ = 0;
  }

  if (exact_face_sum_op && exact_face_sum_op->TimingEnabled())
  {
    exact_face_sum_op->LogTimingSummary(comm,
                                        linear_solve_calls_,
                                        last_linear_iters_,
                                        last_linear_solve_seconds_,
                                        last_precond_name_);
  }

  if (direct_identity_system_)
  {
    X_.SyncAliasMemory(sigma_);
  }
  else
  {
    a_->RecoverFEMSolution(X_, *b_, sigma_);
  }
  if (!last_linear_converged_ &&
      nonconvergence_policy_ == NonconvergencePolicy::Accept)
  {
    MFEM_VERIFY(VectorIsFinite(sigma_),
                "IGR recovery produced non-finite sigma from an accepted "
                "capped linear iterate.");
  }
}

} // namespace dg_igr

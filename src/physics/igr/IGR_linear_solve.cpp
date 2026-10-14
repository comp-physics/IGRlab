#include "igrlab/environment.hpp"
#include "physics/igr/IGR.hpp"

#include "numerics/timestepping/ssprk_coefficients.hpp"
#include "utils/config/env_flags.hpp"
#include "utils/profiling/nvtx.hpp"
#include "IGR_internal_utils.hpp"
#include "IGR_nodal_source.hpp"
#include "IGR_dense_inverse.hpp"
#include "IGR_two_level_inverse.hpp"
#include "IGR_screened_patch_inverse.hpp"
#include "IGR_face_density_cuda.hpp"
#include "IGR_face_density_host_cache.hpp"
#include "IGR_stationary_correction.hpp"
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace igrlab
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
  const char* raw = ::igrlab::GetEnv(name);
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
  const char* raw = ::igrlab::GetEnv(name);
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
  const char* raw = ::igrlab::GetEnv("DG_IGR_IGR_IPDG_FACE_SCALING");
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
        ::igrlab::EnvFlagEnabledDefault(
          "DG_IGR_IGR_CAPTURE_NEGATIVE_QUADRATIC", false))
  {
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::igrlab::NVTXRange range("IGR/krylov-operator-mult");
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
    ::igrlab::NVTXRange range("IGR/krylov-operator-mult-transpose");
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
    const char* prefix = ::igrlab::GetEnv("DG_IGR_IGR_NEGATIVE_CAPTURE_PREFIX");
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
    ::igrlab::NVTXRange range("IGR/krylov-preconditioner-mult");
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
    ::igrlab::NVTXRange range("IGR/krylov-preconditioner-mult-transpose");
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
      nodes_sequence_(sfes.GetMesh() ? sfes.GetMesh()->GetNodesSequence() : -1),
      rule_order_(shared_ir.GetOrder()),
      local_x_gf_(const_cast<mfem::ParFiniteElementSpace*>(&sfes))
  {
    for (int q = 0; q < shared_ir.GetNPoints(); ++q)
      rule_points_.push_back({{shared_ir.IntPoint(q).x, shared_ir.IntPoint(q).weight}});
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
      &scratch_.penalty_metric,
      &scratch_.shape_dictionary);
    timing_enabled_ =
      ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_EXACT_FACE_TIMING",
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
                  const mfem::FaceQuadratureSpace& fqs,
                  const mfem::IntegrationRule& shared_ir,
                  const mfem::Vector& side_q,
                  const mfem::Vector& shared_side_q,
                  mfem::real_t ip_sigma,
                  mfem::real_t kappa,
                  dg_face::DGDiffusionFaceScaling face_scaling) const
  {
    const mfem::Mesh* mesh = sfes.GetMesh();
    return &sfes == &sfes_ && sfes.GetSequence() == sfes_sequence_ && mesh &&
           mesh->GetSequence() == mesh_sequence_ &&
           mesh->GetNodesSequence() == nodes_sequence_ &&
           RuleMatches(fqs.GetIntRule(0)) && RuleMatches(shared_ir) &&
           ip_sigma == ip_sigma_ && kappa == kappa_ &&
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
    ::igrlab::NVTXRange range("IGR/exact-face/Mult");
    if (!timing_enabled_)
    {
      local_y_.SetSize(sfes_.GetVSize(), y);
      local_y_ = 0.0;

      if (prolong_)
      {
        ::igrlab::NVTXRange prolong_range("IGR/exact-face/prolong");
        prolong_->Mult(x, local_x_gf_);
      }
      else
      {
        local_x_gf_ = x;
      }

      {
        ::igrlab::NVTXRange apply_range("IGR/exact-face/local-apply");
        dg_face::LaunchInteriorFaceDGDiffusionApply2D(info_,
                                                      scratch_,
                                                      local_x_gf_,
                                                      ip_sigma_,
                                                      local_y_);
      }
      if (has_shared_)
      {
        {
          ::igrlab::NVTXRange exchange_range("IGR/exact-face/shared-exchange");
          local_x_gf_.ExchangeFaceNbrData();
        }
        {
          ::igrlab::NVTXRange apply_range("IGR/exact-face/shared-apply");
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
        ::igrlab::NVTXRange restrict_range("IGR/exact-face/restrict");
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
      ::igrlab::NVTXRange prolong_range("IGR/exact-face/prolong");
      prolong_->Mult(x, local_x_gf_);
    }
    else
    {
      local_x_gf_ = x;
    }
    timing_.prolong_seconds += ElapsedSeconds(prolong_start);

    const auto local_apply_start = Clock::now();
    {
      ::igrlab::NVTXRange apply_range("IGR/exact-face/local-apply");
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
        ::igrlab::NVTXRange exchange_range("IGR/exact-face/shared-exchange");
        local_x_gf_.ExchangeFaceNbrData();
      }
      timing_.shared_exchange_seconds += ElapsedSeconds(shared_exchange_start);

      const auto shared_apply_start = Clock::now();
      {
        ::igrlab::NVTXRange apply_range("IGR/exact-face/shared-apply");
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
      ::igrlab::NVTXRange restrict_range("IGR/exact-face/restrict");
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

private:
  bool RuleMatches(const mfem::IntegrationRule& ir) const
  {
    if (ir.GetOrder() != rule_order_ ||
        ir.GetNPoints() != static_cast<int>(rule_points_.size())) { return false; }
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const auto& ip = ir.IntPoint(q);
      if (std::memcmp(&ip.x, &rule_points_[q][0], sizeof(ip.x)) ||
          std::memcmp(&ip.weight, &rule_points_[q][1], sizeof(ip.weight))) return false;
    }
    return true;
  }

  const mfem::ParFiniteElementSpace& sfes_;
  const mfem::Operator* prolong_ = nullptr;
  const mfem::Operator* restrict_ = nullptr;
  mfem::real_t ip_sigma_ = 0.0;
  mfem::real_t kappa_ = 0.0;
  dg_face::DGDiffusionFaceScaling face_scaling_ =
    dg_face::DGDiffusionFaceScaling::Arithmetic;
  long sfes_sequence_ = -1;
  long mesh_sequence_ = -1;
  long nodes_sequence_ = -1;
  int rule_order_ = -1;
  std::vector<std::array<mfem::real_t, 2>> rule_points_;
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
    ::igrlab::NVTXRange range("IGR/exact-sum/Mult");
    ++operator_mult_calls_;
    if (!timing_enabled_)
    {
      {
        ::igrlab::NVTXRange base_range("IGR/exact-sum/base");
        base_->Mult(x, y);
      }
      {
        ::igrlab::NVTXRange face_range("IGR/exact-sum/face");
        face_->Mult(x, scratch_);
      }
      {
        ::igrlab::NVTXRange axpy_range("IGR/exact-sum/axpy");
        y += scratch_;
      }
      return;
    }

    const auto base_start = Clock::now();
    {
      ::igrlab::NVTXRange base_range("IGR/exact-sum/base");
      base_->Mult(x, y);
    }
    timing_.base_seconds += ElapsedSeconds(base_start);

    const auto face_start = Clock::now();
    {
      ::igrlab::NVTXRange face_range("IGR/exact-sum/face");
      face_->Mult(x, scratch_);
    }
    timing_.face_seconds += ElapsedSeconds(face_start);

    const auto axpy_start = Clock::now();
    {
      ::igrlab::NVTXRange axpy_range("IGR/exact-sum/axpy");
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
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
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
  const bool exact_face = UseExactDGFaceOperator_();
  if (exact_face)
  {
    MFEM_VERIFY(
      !PreconditionerRequiresAssembledMatrix(prec_type_),
      "The exact side-aware DG face operator requires partial assembly and "
      "is incompatible with an assembled-matrix IGR preconditioner.");
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=partial-assembly-policy"
                << " requested=on"
                << " effective=on"
                << " reason=exact-dg-face-required-partial-assembly"
                << " precond=" << PrecondTypeName(prec_type_) << "\n";
    }
    return true;
  }
  if (device_backend)
  {
    const bool requested =
      ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_PARTIAL_ASSEMBLY", true);
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
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_PARTIAL_ASSEMBLY", false);
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
         ::igrlab::EnvFlagEnabledDefault(
           "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR",
           false);
}

bool IGRSolver::ShouldUseDirectIdentitySystem_(bool use_device_qf,
                                               std::string& reason) const
{
  const bool requested =
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_DIRECT_IDENTITY_SYSTEM",
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
  ::igrlab::NVTXRange range("IGR/assemble-system");
  const auto assemble_start = std::chrono::steady_clock::now();

  // Needed for DG/IPDG assembly: variable coeffs evaluated on shared faces
  // will read U on neighbor elements.
  // For H1 sigma spaces, no DG/IP face terms are assembled, so this exchange is
  // unnecessary. In serial runs, neighbor exchange is also a no-op with extra
  // overhead; skip it in that case as well.
  if (IsDGSpace(sfes_) && mfem::Mpi::WorldSize() > 1)
  {
    ::igrlab::NVTXRange exchange_range("IGR/state-face-neighbor-exchange");
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
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
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
    ::igrlab::NVTXRange coeff_range("IGR/device-coefficients");
    UpdateDeviceCoefficients_(U);
  }
  else
  {
    ::igrlab::NVTXRange coeff_range("IGR/host-coefficients");
    // Ensure host-visible state for coefficient evaluation.
    // The IGR coefficients call GridFunction::GetVectorValue(), which is host-only.
    // On device backends, default to syncing unless explicitly disabled.
    const bool hostsync_default = device_backend;
    if (::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_FORCE_STATE_HOSTSYNC",
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
    ::igrlab::NVTXRange assemble_form_range("IGR/mfem-form-assemble");
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
    ::igrlab::NVTXRange form_range("IGR/form-linear-system");
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
    ::igrlab::NVTXRange exact_setup_range("IGR/exact-face-setup");
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
    const mfem::IntegrationRule& exact_face_ir =
      mfem::IntRules.Get(face_geom, std::max({2 * SpaceOrder(sfes_),
        subcell_quadrature_minimum_order_, OperatorQuadratureMinimumOrder()}));
    if (OperatorQuadratureMinimumOrder() > 0)
    {
      static bool logged_exact_rule = false;
      if (!logged_exact_rule)
      {
        std::cout << "[igr-research] event=exact-face-quadrature requested_min_order="
                  << OperatorQuadratureMinimumOrder()
                  << " actual_order=" << exact_face_ir.GetOrder()
                  << " points=" << exact_face_ir.GetNPoints() << std::endl;
        logged_exact_rule = true;
      }
    }
    const bool use_cuda_face_density =
      ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_FACE_DENSITY_CUDA", false);
    const bool cache_face_density =
      ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_FACE_DENSITY_CACHE", false);
    MFEM_VERIFY(!cache_face_density || use_cuda_face_density,
                "Face-density geometry caching requires the CUDA projection option.");
    std::unique_ptr<mfem::FaceQuadratureSpace> uncached_fqs;
    const mfem::FaceQuadratureSpace* prepared_host_fqs = nullptr;
    bool host_geometry_changed = false;
    if (cache_face_density)
    {
      if (!face_density_cuda_cache_)
        face_density_cuda_cache_ = std::make_shared<FaceDensityCUDAProjectionCache>();
    }
    else if (!use_cuda_face_density)
    {
      if (!face_density_host_cache_)
        face_density_host_cache_ = std::make_shared<HostFaceDensityPreparationCache>();
      prepared_host_fqs = face_density_host_cache_->Prepare(
        vfes_, sfes_, exact_face_ir, &host_geometry_changed);
    }
    if (!cache_face_density && !prepared_host_fqs)
      uncached_fqs = std::make_unique<mfem::FaceQuadratureSpace>(
        *mesh, exact_face_ir, mfem::FaceType::Interior);
    const mfem::FaceQuadratureSpace& exact_fqs = cache_face_density
      ? face_density_cuda_cache_->Prepare(vfes_, sfes_, exact_face_ir)
      : (prepared_host_fqs ? *prepared_host_fqs : *uncached_fqs);
    std::string reason;
    if (!cache_face_density && !prepared_host_fqs && !dg_face::SupportsInteriorFaceSideDensityProjection2D(vfes_,
                                                              exact_fqs,
                                                              &reason))
    {
      last_exact_face_reason_ = "unsupported-interior-side-density";
      MFEM_VERIFY(false,
                  "Unsupported exact IGR side-density face operator: "
                    << reason);
    }
    if (!cache_face_density && !prepared_host_fqs && !dg_face::SupportsInteriorFaceScalarTraceProjection2D(sfes_,
                                                              exact_fqs,
                                                              &reason))
    {
      last_exact_face_reason_ = "unsupported-interior-sigma-trace";
      MFEM_VERIFY(false,
                  "Unsupported exact IGR sigma face operator: " << reason);
    }

    mfem::Vector uncached_side_q;
    dg_face::InteriorFaceSideDensityProjectionScratch uncached_side_scratch;
    mfem::Vector& side_q = cache_face_density ? face_density_cuda_cache_->side_q : uncached_side_q;
    auto& side_scratch = cache_face_density ? face_density_cuda_cache_->work : uncached_side_scratch;
    if (local_alpha)
    {
      MFEM_VERIFY(device_element_alpha_.Size() == vfes_.GetNE(),
                  "Exact local-alpha face projection requires the current "
                  "device element-alpha cache.");
    }
    dg_face::InteriorFaceSideDensityProjectionInfo side_info;
    if (use_cuda_face_density)
    {
#if defined(DG_IGR_HAS_FACE_DENSITY_CUDA)
      MFEM_VERIFY(mfem::Device::Allows(mfem::Backend::CUDA),
                  "CUDA face-density projection requires the CUDA backend.");
      side_info = ProjectInteriorFaceDensityCoefficientCUDA(
        vfes_, exact_fqs, U, alpha_, local_alpha ? &device_element_alpha_ : nullptr,
        rho_floor_, side_q, side_scratch,
        cache_face_density ? &face_density_cuda_cache_->info : nullptr);
#else
      MFEM_ABORT("CUDA face-density projection requires a CUDA build.");
#endif
    }
    else if (prepared_host_fqs)
      side_info = face_density_host_cache_->Project(
        vfes_, U, alpha_, local_alpha ? &device_element_alpha_ : nullptr,
        rho_floor_, side_q);
    else
      side_info = local_alpha
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
    if (linear_solve_calls_ == 0 && mfem::Mpi::Root())
      std::cout << "[igr-linear] event=face-density-path arithmetic="
                << (use_cuda_face_density ? "cuda" : "host")
                << " scope=local-interior-faces geometry_cache="
                << (cache_face_density ? "per-solver"
                    : prepared_host_fqs ? "host-per-solver" : "none") << std::endl;
    if (cache_face_density &&
        (face_density_cuda_cache_->hits == 0 || debug_qf))
      std::cout << "[igr-linear] event=face-density-cache rebuilds="
                << face_density_cuda_cache_->rebuilds << " hits=" << face_density_cuda_cache_->hits
                << " shape_entries=" << face_density_cuda_cache_->work.shape.Size()
                << " map_entries=" << face_density_cuda_cache_->work.element_vdof_map.Size()
                << std::endl;
    last_exact_face_local_nf_ = side_info.nf;
    const bool ingestion_audit =
      ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_INGESTION_AUDIT", false);
    if (ingestion_audit)
    {
      const mfem::Vector* audited_side_alpha = local_alpha
        ? (prepared_host_fqs ? &face_density_host_cache_->CurrentSideAlpha()
                             : &side_scratch.side_alpha) : nullptr;
      LogFaceCoefficientAudit("local",
                              side_q,
                              alpha_,
                              rho_floor_,
                              linear_solve_calls_ + 1,
                              vfes_.GetComm(),
                              audited_side_alpha,
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
      sigma_cadence_ == IGRSigmaUpdateCadence::ExtrapolatedEveryStep
        ? dg_face::DGDiffusionFaceScaling::Arithmetic
        : ReadIPDGFaceScaling();
    face_op = dynamic_cast<ExactL2DGFaceOperator*>(
      exact_face_operator_.get());
    const bool reused_face_geometry =
      !host_geometry_changed && face_op && face_op->CanRefresh(sfes_,
                                     exact_fqs,
                                     exact_face_ir,
                                     side_q,
                                     shared_side_q,
                                     static_cast<mfem::real_t>(ip_kind_),
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
              << (::igrlab::EnvFlagEnabledDefault(
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
  ::igrlab::NVTXRange range("IGR/smoother-sweeps");
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
  ::igrlab::NVTXRange range("IGR/solve-elliptic-system");
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
    ::igrlab::GetEnv("DG_IGR_IGR_EVENT_CAPTURE_PREFIX");
  const bool event_capture_enabled =
    event_capture_prefix_text && event_capture_prefix_text[0] != '\0';
  const int event_capture_min_iters =
    ReadEnvInt("DG_IGR_IGR_EVENT_CAPTURE_MIN_ITERS",
               event_log_min_iters > 0 ? event_log_min_iters : lin_maxit_);
  const bool true_residual_requested =
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_TRUE_RESIDUAL", false);
  const bool true_initial_residual_requested =
    true_residual_requested && ::igrlab::EnvFlagEnabledDefault(
                                 "DG_IGR_IGR_TRUE_INITIAL_RESIDUAL", false);
  const int extended_certificate_start =
    ReadEnvInt("DG_IGR_IGR_EXTENDED_CERTIFICATE_START", 1000);
  const double extended_certificate_factor =
    ReadEnvReal("DG_IGR_IGR_EXTENDED_CERTIFICATE_FACTOR", 5.0);
  MFEM_VERIFY(extended_certificate_start > 0,
              "DG_IGR_IGR_EXTENDED_CERTIFICATE_START must be positive.");
  MFEM_VERIFY(std::isfinite(extended_certificate_factor) &&
                extended_certificate_factor > 0.0,
              "DG_IGR_IGR_EXTENDED_CERTIFICATE_FACTOR must be finite and "
              "positive.");
  const bool extended_certificate_enabled =
    ::igrlab::EnvFlagEnabledDefault(
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
  const char* dense_inverse_env = ::igrlab::GetEnv("DG_IGR_IGR_DENSE_INVERSE");
  const std::string dense_inverse_precision =
    dense_inverse_env ? dense_inverse_env : "";
  const bool using_dense_inverse = !dense_inverse_precision.empty();
  const char* two_level_env = ::igrlab::GetEnv("DG_IGR_IGR_TWO_LEVEL_INVERSE");
  const bool using_two_level_inverse = two_level_env && two_level_env[0];
  const char* patch_env = ::igrlab::GetEnv("DG_IGR_IGR_PATCH_RADIUS");
  const bool using_screened_patch = patch_env && patch_env[0];
  const int patch_radius = using_screened_patch
    ? ReadEnvInt("DG_IGR_IGR_PATCH_RADIUS", -1) : -1;
  const char* patch_precision_env = ::igrlab::GetEnv("DG_IGR_IGR_PATCH_PRECISION");
  const std::string patch_precision = using_screened_patch && patch_precision_env
    ? patch_precision_env : "fp64";
  const char* patch_kind_env = ::igrlab::GetEnv("DG_IGR_IGR_PATCH_KIND");
  const std::string patch_kind = using_screened_patch && patch_kind_env
    ? patch_kind_env : "ras";
  const int patch_rebuild_interval = using_screened_patch
    ? ReadEnvInt("DG_IGR_IGR_PATCH_REBUILD_INTERVAL", 0) : 0;
  const std::string patch_key = std::to_string(patch_radius) + ":" + patch_precision +
    ":" + std::to_string(patch_rebuild_interval) + ":" + patch_kind;
  MFEM_VERIFY(!using_screened_patch ||
                (using_cuda && !using_dense_inverse && !using_two_level_inverse &&
                 patch_radius >= 0 && patch_radius <= 6 &&
                 patch_rebuild_interval >= 0 &&
                 (patch_precision == "fp64" || patch_precision == "fp32" || patch_precision == "tf32") &&
                 (patch_kind == "ras" || ((patch_kind == "face-condensed" || patch_kind == "trace-pairs") && patch_radius == 0 && spd_ok) || (patch_kind == "vertex-p1" &&
                    patch_radius == 0 && patch_precision == "fp64" && spd_ok))),
              "Screened patch inverse requires CUDA, radius 0..6, fp64|fp32|tf32, "
              "a nonnegative rebuild interval, "
              "and no global dense or two-level inverse. kind=vertex-p1 additionally "
              "requires radius=0, fp64 and symmetric SIPG; face-condensed requires radius=0 and symmetric SIPG.");
  const char* two_level_coarse_env = ::igrlab::GetEnv("DG_IGR_IGR_TWO_LEVEL_COARSE");
  const std::string two_level_coarse =
    two_level_coarse_env ? two_level_coarse_env : "dense";
  const int two_level_order = using_two_level_inverse
    ? ReadEnvInt("DG_IGR_IGR_TWO_LEVEL_ORDER", 1) : 1;
  const std::string two_level_key = two_level_coarse + ":" + std::to_string(two_level_order);
  MFEM_VERIFY(!using_two_level_inverse ||
                (std::string(two_level_env) == "fp64" && using_cuda &&
                 !using_dense_inverse),
              "Two-level inverse requires CUDA, fp64 and no global dense inverse.");
  MFEM_VERIFY(!using_two_level_inverse ||
                two_level_coarse == "dense" || two_level_coarse == "cg",
              "DG_IGR_IGR_TWO_LEVEL_COARSE must be dense|cg.");
  MFEM_VERIFY(!using_dense_inverse || dense_inverse_precision == "fp64" ||
                dense_inverse_precision == "fp32" ||
                dense_inverse_precision == "tf32",
              "DG_IGR_IGR_DENSE_INVERSE must be fp64|fp32|tf32 or unset.");
  MFEM_VERIFY(!using_dense_inverse || using_cuda,
              "DG_IGR_IGR_DENSE_INVERSE requires the CUDA backend.");
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
  MFEM_VERIFY(!using_dense_inverse || dense_inverse_precision == "fp64" ||
                solver_choice == LinearSolverType::FGMRES,
              "Mixed-precision dense inverse requires explicit FGMRES.");
  MFEM_VERIFY(!using_two_level_inverse ||
                solver_choice == LinearSolverType::FGMRES,
              "Two-level inverse experiment requires explicit FGMRES.");
  MFEM_VERIFY(!using_screened_patch || solver_choice == LinearSolverType::FGMRES ||
                ((patch_kind == "vertex-p1" || patch_kind == "face-condensed" || patch_kind == "trace-pairs") &&
                 patch_precision == "fp64" && solver_choice == LinearSolverType::CG),
              "Screened RAS and mixed precision require FGMRES; FP64 vertex/face-condensed also permit CG.");

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
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_KRYLOV_APPLY_TIMING",
                                    false);
  krylov_telemetry.sync =
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_KRYLOV_APPLY_SYNC",
                                    false);
  const bool use_krylov_telemetry =
    krylov_telemetry.record_seconds ||
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_KRYLOV_APPLY_COUNTS",
                                    true);

  const auto pc_setup_start = std::chrono::steady_clock::now();
  std::unique_ptr<KrylovTelemetryOperator> cheb_operator_telemetry;
  std::unique_ptr<mfem::Solver> owned_prec;
  mfem::Solver* prec = nullptr;
  bool using_operator_jacobi = false;
  bool using_operator_jacobi_exact_face_diag = false;
  bool using_operator_chebyshev = false;
  bool using_operator_chebyshev_base = false;
  {
    ::igrlab::NVTXRange pc_range("IGR/preconditioner-setup");
    if (using_screened_patch)
    {
#if defined(DG_IGR_HAS_SCREENED_PATCH_CUDA)
      MFEM_VERIFY(!screened_patch_preconditioner_ || screened_patch_key_ == patch_key,
                  "Cannot change screened patch configuration on a live solver.");
      if (!screened_patch_preconditioner_ ||
          (patch_rebuild_interval > 0 && (solve_call - 1) % patch_rebuild_interval == 0))
      {
        screened_patch_preconditioner_ = MakeScreenedPatchInversePreconditioner(
          *A_op, sfes_, patch_radius, patch_precision, patch_kind);
        screened_patch_key_ = patch_key;
        if (patch_rebuild_interval > 0 && mfem::Mpi::WorldRank() == 0)
        {
          std::cout << "[igr-screened-patch] event=rebuild solve_call=" << solve_call
                    << " interval=" << patch_rebuild_interval << std::endl;
        }
      }
      screened_patch_preconditioner_->SetOperator(*A_op);
      prec = screened_patch_preconditioner_.get();
#else
      MFEM_ABORT("Screened patch inverse requires a CUDA build.");
#endif
    }
    else if (using_two_level_inverse)
    {
#if defined(DG_IGR_HAS_TWO_LEVEL_INVERSE_CUDA)
      if (!two_level_inverse_preconditioner_)
      {
        two_level_inverse_preconditioner_ =
          MakeTwoLevelInversePreconditioner(*A_op, sfes_, two_level_coarse, two_level_order);
        two_level_coarse_mode_ = two_level_key;
      }
      MFEM_VERIFY(two_level_coarse_mode_ == two_level_key,
                  "Cannot change two-level coarse mode on a live solver.");
      two_level_inverse_preconditioner_->SetOperator(*A_op);
      prec = two_level_inverse_preconditioner_.get();
#else
      MFEM_ABORT("Two-level inverse experiment requires a CUDA build.");
#endif
    }
    else if (using_dense_inverse)
    {
#if defined(DG_IGR_HAS_DENSE_INVERSE_CUDA)
      if (!dense_inverse_preconditioner_)
      {
        dense_inverse_preconditioner_ =
          MakeDenseInversePreconditioner(*A_op, dense_inverse_precision, comm);
        dense_inverse_precision_ = dense_inverse_precision;
      }
      MFEM_VERIFY(dense_inverse_precision_ == dense_inverse_precision,
                  "Cannot change dense inverse precision on a live solver.");
      dense_inverse_preconditioner_->SetOperator(*A_op);
      prec = dense_inverse_preconditioner_.get();
#else
      MFEM_ABORT("Dense inverse experiment requires a CUDA build.");
#endif
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
      owned_prec = std::move(amg);
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
      owned_prec = std::move(ilu);
    }
    else if (prec_choice == PrecondType::HypreEuclid)
    {
      MFEM_VERIFY(A_hypre != nullptr,
                  "HypreEuclid preconditioner requires a Hypre matrix operator.");
      auto euclid = std::make_unique<mfem::HypreEuclid>(*A_hypre);
      euclid->SetStats(0);
      owned_prec = std::move(euclid);
    }
    else if (prec_choice == PrecondType::ParaSails)
    {
      MFEM_VERIFY(A_hypre != nullptr,
                  "ParaSails preconditioner requires a Hypre matrix operator.");
      auto parasails = std::make_unique<mfem::HypreParaSails>(*A_hypre);
      parasails->SetLogging(0);
      owned_prec = std::move(parasails);
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
        owned_prec = std::move(sm);
      }
      else
      {
        const bool use_exact_face_diag =
          exact_face_sum_op &&
          ::igrlab::EnvFlagEnabledDefault(
            "DG_IGR_IGR_EXACT_FACE_DIAGONAL_PRECOND",
            false);
        if (use_exact_face_diag)
        {
          owned_prec = MakePositiveOperatorJacobi(*A_op, ess_tdof_list_);
          using_operator_jacobi_exact_face_diag = true;
        }
        else
        {
          owned_prec = MakePositiveOperatorJacobi(*a_, ess_tdof_list_);
        }
        using_operator_jacobi = true;
      }
    }
    else if (prec_choice == PrecondType::OperatorChebyshev)
    {
      ::igrlab::NVTXRange cheb_range("IGR/chebyshev-setup");
      // The qualified extrapolated instantaneous-IGR route uses the complete
      // exact-face operator for both Krylov and Chebyshev applications.  Do
      // not let the legacy experimental environment knob silently substitute
      // the volume/base operator in that typed temporal discretization.
      const bool use_base_operator =
        exact_face_sum_op &&
        sigma_cadence_ != IGRSigmaUpdateCadence::ExtrapolatedEveryStep &&
        ::igrlab::EnvFlagEnabledDefault(
          "DG_IGR_IGR_CHEB_BASE_OPERATOR",
          false);
      mfem::Operator* cheb_op =
        use_base_operator
          ? &exact_face_sum_op->BaseOperator()
          : A_op;
      mfem::Vector diag(cheb_op->Height());
      const auto cheb_diag_start = Clock::now();
      {
        ::igrlab::NVTXRange diag_range("IGR/chebyshev-diagonal");
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
           ::igrlab::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false)) &&
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
      owned_prec = std::move(cheb);
      using_operator_chebyshev = true;
      using_operator_chebyshev_base = use_base_operator;
    }
    // else None -> leave prec null
    if (!using_dense_inverse && !using_two_level_inverse && !using_screened_patch)
    { prec = owned_prec.get(); }
  }
  const auto pc_setup_stop = std::chrono::steady_clock::now();
  last_pc_setup_seconds_ =
    std::chrono::duration<double>(pc_setup_stop - pc_setup_start).count();
  total_pc_setup_seconds_ += last_pc_setup_seconds_;
  if (run_operator_probe && prec &&
      ::igrlab::EnvFlagEnabledDefault(
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
      ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_WARM_START_GUARD", false))
  {
    ::igrlab::NVTXRange warm_guard_range("IGR/warm-start-residual-guard");
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

  const char* stationary_env = ::igrlab::GetEnv("DG_IGR_IGR_STATIONARY_SWEEPS");
  int stationary_sweeps = 0;
  if (stationary_env && stationary_env[0])
  {
    char* end = nullptr;
    const long requested = std::strtol(stationary_env, &end, 10);
    MFEM_VERIFY(end && *end == '\0' && requested >= 1 && requested <= 16 &&
                  using_cuda && solver_choice == LinearSolverType::FGMRES &&
                  solve_preconditioner,
                "Stationary correction requires CUDA, FGMRES, a preconditioner, and 1..16 sweeps.");
    stationary_sweeps = static_cast<int>(requested);
#ifdef DG_IGR_HAS_STATIONARY_CORRECTION_CUDA
    if (!stationary_correction_)
    {
      stationary_correction_ = MakeStationaryCorrectionSolver(comm, stationary_sweeps);
      stationary_sweeps_ = stationary_sweeps;
      if (mfem::Mpi::Root())
      {
        std::cout << "[igr-stationary] event=setup sweeps=" << stationary_sweeps
                  << " damping=1 precision=fp64 gate=current-true-residual fallback=fgmres"
                  << std::endl;
      }
    }
    MFEM_VERIFY(stationary_sweeps_ == stationary_sweeps,
                "Stationary sweep count cannot change during a solver lifetime.");
#else
    MFEM_ABORT("Stationary correction requires the CUDA implementation.");
#endif
  }

  const auto solve_start = std::chrono::steady_clock::now();

  auto run_krylov = [&](double relative_tolerance, bool iterative_mode)
  {
    ::igrlab::NVTXRange krylov_range("IGR/krylov-solve");
    if (stationary_sweeps)
    {
      configure_solver(*stationary_correction_, relative_tolerance, iterative_mode);
      stationary_correction_->Mult(B_, X_);
      if (stationary_correction_->GetConverged())
      {
        last_linear_iters_ = stationary_correction_->GetNumIterations();
        last_linear_converged_ = true;
        last_linear_final_norm_ = stationary_correction_->GetFinalNorm();
        last_linear_initial_norm_ = stationary_correction_->GetInitialNorm();
        return;
      }
      // The rejected trial restores X_; charge both trial and Krylov work.
    }
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
    last_linear_iters_ += stationary_sweeps;
  };

  run_krylov(effective_rtol, enable_warm_start);

  const bool certify_extended_solve =
    extended_certificate_enabled &&
    last_linear_iters_ >= extended_certificate_start;
  const bool certify_independent_residual =
    certify_current_solve_ || certify_extended_solve;
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
  if (true_residual_requested || certify_independent_residual)
  {
    recompute_true_residual();
  }

  bool independent_certificate_passed = true;
  int residual_replacement_retries = 0;
  if (certify_independent_residual)
  {
    // A caller-requested production certificate is a fixed part of the
    // method and must not be weakened by the optional diagnostic environment
    // used to screen unusually long solves.
    constexpr double caller_certificate_factor = 5.0;
    const double certificate_factor = certify_current_solve_
                                        ? caller_certificate_factor
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
    const bool successful_production_anchor =
      independent_certificate_passed &&
      current_solve_purpose_ != IGRSolvePurpose::OutputOnly &&
      rk_stage_index_ == -1;
    if (mfem::Mpi::Root() && !successful_production_anchor)
    {
      std::cout << "[igr-linear] event=independent-residual-certificate"
                << " solve_call=" << solve_call
                << " rk_step=" << rk_step_index_
                << " rk_stage=" << rk_stage_index_
                << " iterations=" << last_linear_iters_
                << " reason="
                << (certify_current_solve_ ? "caller-requested"
                                           : "extended-solve")
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
  last_precond_name_ = using_operator_chebyshev
                         ? (using_operator_chebyshev_base
                              ? "operatorchebyshev-base"
                              : "operatorchebyshev")
                         : (using_operator_jacobi_exact_face_diag
                              ? "operatorjacobi-exactface"
                              : (using_operator_jacobi
                                   ? "operatorjacobi"
                                   : PrecondTypeName(prec_choice)));
  if (using_dense_inverse)
  {
    last_precond_name_ = "dense-inverse-" + dense_inverse_precision;
  }
  if (using_two_level_inverse)
  {
    last_precond_name_ = "two-level-inverse-fp64";
  }
  if (using_screened_patch)
  {
    last_precond_name_ = patch_kind == "ras" ? "screened-patch-" + patch_precision
                                            : "screened-" + patch_kind + "-" + patch_precision;
  }

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

} // namespace igrlab

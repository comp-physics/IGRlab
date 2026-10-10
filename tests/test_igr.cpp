#include "igrlab/environment.hpp"
#include "physics/euler/conversions.hpp"
#include "physics/igr/IGR.hpp"
#include "physics/igr/igr_coupling.hpp"
#include "physics/igr/igr_flux.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/riemann/side_scope_utils.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "sim/io/vis_helpers.hpp"
#include "../src/numerics/dg/DGFaceQuadratureKernels.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace dg_igr_tests
{

namespace
{

mfem::Mesh MakePeriodicMesh1D(int n)
{
  mfem::Mesh orig_mesh = mfem::Mesh::MakeCartesian1D(n, 1.0);
  std::vector<mfem::Vector> translations = {mfem::Vector({1.0})};
  return mfem::Mesh::MakePeriodic(
    orig_mesh,
    orig_mesh.CreatePeriodicVertexMapping(translations));
}

mfem::Mesh MakePeriodicMesh2D(int nx, int ny, double sx = 1.0, double sy = 1.0)
{
  mfem::Mesh orig_mesh =
    mfem::Mesh::MakeCartesian2D(nx,
                                ny,
                                mfem::Element::QUADRILATERAL,
                                /*generate_edges=*/true,
                                sx,
                                sy);
  mfem::Vector shift_x(2);
  shift_x(0) = sx;
  shift_x(1) = 0.0;
  mfem::Vector shift_y(2);
  shift_y(0) = 0.0;
  shift_y(1) = sy;
  std::vector<mfem::Vector> translations = {shift_x, shift_y};
  return mfem::Mesh::MakePeriodic(
    orig_mesh,
    orig_mesh.CreatePeriodicVertexMapping(translations));
}

mfem::Mesh MakeGradedQuadMesh2D(int nx = 4,
                                int ny = 2,
                                double sx = 1.0,
                                double sy = 0.6)
{
  mfem::Mesh mesh =
    mfem::Mesh::MakeCartesian2D(nx,
                                ny,
                                mfem::Element::QUADRILATERAL,
                                /*generate_edges=*/true,
                                sx,
                                sy);
  // Monotone tensor-product grading.  Grading both coordinates makes a
  // two-rank partition encounter unequal element sizes whether MFEM cuts the
  // Cartesian mesh by rows or columns.
  for (int v = 0; v < mesh.GetNV(); ++v)
  {
    double* x = mesh.GetVertex(v);
    const double tx = x[0] / sx;
    const double ty = x[1] / sy;
    x[0] = sx * (0.18 * tx + 0.82 * tx * tx);
    x[1] = sy * (0.18 * ty + 0.82 * ty * ty);
  }
  return mesh;
}

mfem::FaceElementTransformations& GetFaceTr(mfem::ParMesh& pmesh, int face)
{
  mfem::FaceElementTransformations* Tr =
    pmesh.GetFaceElementTransformations(face);
  MFEM_VERIFY(Tr != nullptr, "Failed to obtain face transformations.");
  const mfem::IntegrationRule& ir =
    mfem::IntRules.Get(Tr->GetGeometryType(), /*order=*/1);
  const mfem::IntegrationPoint& ip = ir.IntPoint(0);
  Tr->SetAllIntPoints(&ip);
  return *Tr;
}

class StepSigmaCoefficient final : public mfem::Coefficient
{
public:
  StepSigmaCoefficient(double left, double right, double split_x)
    : left_(left),
      right_(right),
      split_x_(split_x)
  {
  }

  double Eval(mfem::ElementTransformation& Tr,
              const mfem::IntegrationPoint& ip) override
  {
    mfem::Vector x(Tr.GetSpaceDim());
    Tr.Transform(ip, x);
    return (x(0) < split_x_) ? left_ : right_;
  }

private:
  double left_ = 0.0;
  double right_ = 0.0;
  double split_x_ = 0.5;
};

class AlphaOverRhoCoefficient final : public mfem::Coefficient
{
public:
  AlphaOverRhoCoefficient(const mfem::ParGridFunction& U,
                          double alpha,
                          double rho_floor)
    : U_(&U),
      alpha_(alpha),
      rho_floor_(rho_floor)
  {
  }

  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
  {
    MFEM_VERIFY(U_, "AlphaOverRhoCoefficient requires a valid state.");
    U_->GetVectorValue(T, ip, state_);
    const double rho = std::max(state_(0), rho_floor_);
    return alpha_ / rho;
  }

private:
  const mfem::ParGridFunction* U_ = nullptr;
  double alpha_ = 0.0;
  double rho_floor_ = 0.0;
  mfem::Vector state_;
};

class InverseRhoCoefficient final : public mfem::Coefficient
{
public:
  InverseRhoCoefficient(const mfem::ParGridFunction& U, double rho_floor)
    : U_(&U),
      rho_floor_(rho_floor)
  {
  }

  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
  {
    MFEM_VERIFY(U_, "InverseRhoCoefficient requires a valid state.");
    U_->GetVectorValue(T, ip, state_);
    const double rho = std::max(state_(0), rho_floor_);
    return 1.0 / rho;
  }

private:
  const mfem::ParGridFunction* U_ = nullptr;
  double rho_floor_ = 0.0;
  mfem::Vector state_;
};

/// Test-only DG shell whose RHS contributes controlled cumulative q telemetry.
class SyntheticQSigmaTelemetryDG final
  : public mfem::DGHyperbolicConservationLaws
{
public:
  using mfem::DGHyperbolicConservationLaws::DGHyperbolicConservationLaws;

  void Mult(const mfem::Vector&, mfem::Vector& derivative) const override
  {
    derivative = 0.0;
    telemetry_.enabled = true;
    telemetry_.sample_count += 8;
    if (emit_failure_)
    {
      telemetry_.nonpositive_count += next_nonpositive_;
      telemetry_.nonfinite_count += next_nonfinite_;
      emit_failure_ = false;
    }
  }

  mfem::IGRFrozenSigmaTelemetry GetGlobalFrozenSigmaTelemetry() const override
  {
    return telemetry_;
  }

  mfem::IGRFrozenSigmaTelemetry InspectFrozenSigmaStateForAdmission(
    const mfem::Vector&) const override
  {
    mfem::IGRFrozenSigmaTelemetry fresh;
    fresh.enabled = true;
    fresh.sample_count = 8;
    if (emit_endpoint_failure_)
    {
      fresh.nonpositive_count = next_endpoint_nonpositive_;
      fresh.nonfinite_count = next_endpoint_nonfinite_;
      emit_endpoint_failure_ = false;
    }
    telemetry_.Merge(fresh);
    return fresh;
  }

  void SeedHistoricalFailure(std::uint64_t nonpositive,
                             std::uint64_t nonfinite)
  {
    telemetry_.enabled = true;
    telemetry_.sample_count += nonpositive + nonfinite;
    telemetry_.nonpositive_count += nonpositive;
    telemetry_.nonfinite_count += nonfinite;
  }

  void FailNextRHS(std::uint64_t nonpositive, std::uint64_t nonfinite)
  {
    emit_failure_ = true;
    next_nonpositive_ = nonpositive;
    next_nonfinite_ = nonfinite;
  }

  void FailNextEndpointAdmission(std::uint64_t nonpositive,
                                 std::uint64_t nonfinite)
  {
    emit_endpoint_failure_ = true;
    next_endpoint_nonpositive_ = nonpositive;
    next_endpoint_nonfinite_ = nonfinite;
  }

private:
  mutable mfem::IGRFrozenSigmaTelemetry telemetry_;
  mutable bool emit_failure_ = false;
  mutable std::uint64_t next_nonpositive_ = 0;
  mutable std::uint64_t next_nonfinite_ = 0;
  mutable bool emit_endpoint_failure_ = false;
  mutable std::uint64_t next_endpoint_nonpositive_ = 0;
  mutable std::uint64_t next_endpoint_nonfinite_ = 0;
};

class BrokenDGIGRSourceCoefficient final : public mfem::Coefficient
{
public:
  BrokenDGIGRSourceCoefficient(const mfem::ParGridFunction& U,
                               int dim,
                               int num_eq,
                               double alpha,
                               double rho_floor)
    : U_(&U),
      dim_(dim),
      num_eq_(num_eq),
      alpha_(alpha),
      rho_floor_(rho_floor),
      state_(num_eq),
      gradU_(num_eq, dim),
      gradu_(dim, dim)
  {
  }

  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
  {
    MFEM_VERIFY(U_, "BrokenDGIGRSourceCoefficient requires a valid state.");
    T.SetIntPoint(&ip);
    U_->GetVectorValue(T, ip, state_);
    U_->GetVectorGradient(T, gradU_);
    return EvalFromStateAndGradient();
  }

  double EvalFromStateAndGradient()
  {
    const double rho = std::max(state_(0), rho_floor_);
    const double rho2 = rho * rho;
    for (int i = 0; i < dim_; ++i)
    {
      for (int j = 0; j < dim_; ++j)
      {
        const double mom_i = state_(1 + i);
        const double dmij = gradU_(1 + i, j);
        const double drhoj = gradU_(0, j);
        gradu_(i, j) = (rho * dmij - mom_i * drhoj) / rho2;
      }
    }

    double divu = 0.0;
    for (int i = 0; i < dim_; ++i)
    {
      divu += gradu_(i, i);
    }

    double trA2 = 0.0;
    for (int i = 0; i < dim_; ++i)
    {
      for (int j = 0; j < dim_; ++j)
      {
        trA2 += gradu_(i, j) * gradu_(j, i);
      }
    }
    return alpha_ * (trA2 + divu * divu);
  }

  mfem::Vector& StateScratch() { return state_; }
  mfem::DenseMatrix& GradientScratch() { return gradU_; }

private:
  const mfem::ParGridFunction* U_ = nullptr;
  int dim_ = 0;
  int num_eq_ = 0;
  double alpha_ = 0.0;
  double rho_floor_ = 0.0;
  mfem::Vector state_;
  mfem::DenseMatrix gradU_;
  mfem::DenseMatrix gradu_;
};

struct ExactFaceApplyCheckResult
{
  double rel_err = 0.0;
  double max_abs_err = 0.0;
  double ref_norm = 0.0;
  int nf = 0;
  int nq = 0;
};

ExactFaceApplyCheckResult CompareExactFaceApplyWithAssembledMFEM(
  mfem::ParMesh& pmesh,
  mfem::ParFiniteElementSpace& state_fes,
  mfem::ParFiniteElementSpace& sigma_fes,
  mfem::ParGridFunction& state,
  mfem::ParGridFunction& sigma,
  const mfem::IntegrationRule& face_ir,
  double alpha,
  double rho_floor,
  double ip_sigma,
  double kappa)
{
  mfem::FaceQuadratureSpace fqs(pmesh,
                                face_ir,
                                mfem::FaceType::Interior);

  mfem::Vector side_q;
  igrlab::dg_face::InteriorFaceSideDensityProjectionScratch coeff_scratch;
  const auto coeff_info =
    igrlab::dg_face::ProjectInteriorFaceDensityCoefficient2D(state_fes,
                                                             fqs,
                                                             state,
                                                             alpha,
                                                             rho_floor,
                                                             side_q,
                                                             &coeff_scratch);

  mfem::Vector exact_face_y(sigma.Size());
  exact_face_y = 0.0;
  igrlab::dg_face::InteriorFaceDGDiffusionApplyScratch apply_scratch;
  const auto apply_info =
    igrlab::dg_face::AddInteriorFaceDGDiffusionApply2D(sigma_fes,
                                                       fqs,
                                                       sigma,
                                                       side_q,
                                                       ip_sigma,
                                                       kappa,
                                                       exact_face_y,
                                                       &apply_scratch);

  MFEM_VERIFY(coeff_info.qf_size == 2 * apply_info.nq * apply_info.nf,
              "Exact face apply comparison received mismatched qdata.");

  AlphaOverRhoCoefficient alpha_over_rho(state, alpha, rho_floor);
  mfem::DGDiffusionIntegrator mfem_face(alpha_over_rho, ip_sigma, kappa);
  mfem_face.SetIntegrationRule(face_ir);

  mfem::Vector assembled_face_y(sigma.Size());
  assembled_face_y = 0.0;
  const double* sigma_data = sigma.HostRead();
  double* assembled_data = assembled_face_y.HostReadWrite();
  const int sigma_dof = sigma_fes.GetFE(0)->GetDof();
  mfem::Array<int> sigma_dofs;
  mfem::DenseMatrix elmat;
  mfem::Vector xloc(2 * sigma_dof);
  mfem::Vector yloc(2 * sigma_dof);

  for (int f = 0; f < fqs.GetNE(); ++f)
  {
    const int mesh_face = fqs.GetMeshFaceIndex(f);
    mfem::FaceElementTransformations* Tr =
      pmesh.GetFaceElementTransformations(mesh_face);
    MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                "Exact face apply comparison expected interior face.");
    const mfem::FiniteElement& fe1 = *sigma_fes.GetFE(Tr->Elem1No);
    const mfem::FiniteElement& fe2 = *sigma_fes.GetFE(Tr->Elem2No);
    mfem_face.AssembleFaceMatrix(fe1, fe2, *Tr, elmat);

    sigma_fes.GetElementDofs(Tr->Elem1No, sigma_dofs);
    MFEM_VERIFY(sigma_dofs.Size() == sigma_dof,
                "Exact face apply comparison side-1 dof mismatch.");
    for (int d = 0; d < sigma_dof; ++d)
    {
      int vdof = sigma_fes.DofToVDof(sigma_dofs[d], 0);
      vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
      xloc(d) = sigma_data[vdof];
    }

    sigma_fes.GetElementDofs(Tr->Elem2No, sigma_dofs);
    MFEM_VERIFY(sigma_dofs.Size() == sigma_dof,
                "Exact face apply comparison side-2 dof mismatch.");
    for (int d = 0; d < sigma_dof; ++d)
    {
      int vdof = sigma_fes.DofToVDof(sigma_dofs[d], 0);
      vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
      xloc(sigma_dof + d) = sigma_data[vdof];
    }

    elmat.Mult(xloc, yloc);

    sigma_fes.GetElementDofs(Tr->Elem1No, sigma_dofs);
    for (int d = 0; d < sigma_dof; ++d)
    {
      int vdof = sigma_fes.DofToVDof(sigma_dofs[d], 0);
      vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
      assembled_data[vdof] += yloc(d);
    }

    sigma_fes.GetElementDofs(Tr->Elem2No, sigma_dofs);
    for (int d = 0; d < sigma_dof; ++d)
    {
      int vdof = sigma_fes.DofToVDof(sigma_dofs[d], 0);
      vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
      assembled_data[vdof] += yloc(sigma_dof + d);
    }
  }

  mfem::Vector face_delta(exact_face_y);
  face_delta -= assembled_face_y;
  ExactFaceApplyCheckResult result;
  result.ref_norm = std::max(GlobalL2Norm(assembled_face_y), 1.0e-30);
  result.rel_err = GlobalL2Norm(face_delta) / result.ref_norm;
  const double* delta_data = face_delta.HostRead();
  for (int i = 0; i < face_delta.Size(); ++i)
  {
    result.max_abs_err = std::max(result.max_abs_err, std::abs(delta_data[i]));
  }
  result.nf = apply_info.nf;
  result.nq = apply_info.nq;
  return result;
}

class LinearAverageFlux final : public mfem::FluxFunction
{
public:
  LinearAverageFlux(int dim, const mfem::Vector& dir)
    : mfem::FluxFunction(dim + 2, dim),
      dir_(dir)
  {
    MFEM_VERIFY(dir_.Size() == dim,
                "LinearAverageFlux direction dimension mismatch.");
  }

  mfem::real_t ComputeFlux(const mfem::Vector& state,
                           mfem::ElementTransformation&,
                           mfem::DenseMatrix& flux) const override
  {
    flux.SetSize(num_equations, dim);
    for (int d = 0; d < dim; ++d)
    {
      for (int i = 0; i < num_equations; ++i)
      {
        flux(i, d) = dir_(d) * state(i);
      }
    }
    return dir_.Norml2();
  }

  mfem::real_t ComputeAvgFlux(const mfem::Vector& state1,
                              const mfem::Vector& state2,
                              mfem::ElementTransformation& Tr,
                              mfem::DenseMatrix& flux) const override
  {
    mfem::Vector avg(state1);
    avg += state2;
    avg *= 0.5;
    return ComputeFlux(avg, Tr, flux);
  }

  void ComputeFluxJacobian(const mfem::Vector&,
                           mfem::ElementTransformation&,
                           mfem::DenseTensor& jacobian) const override
  {
    jacobian.SetSize(num_equations, num_equations, dim);
    for (int d = 0; d < dim; ++d)
    {
      for (int equation = 0; equation < num_equations; ++equation)
      {
        for (int state_component = 0; state_component < num_equations;
             ++state_component)
        {
          jacobian(equation, state_component, d) =
            equation == state_component ? dir_(d) : 0.0;
        }
      }
    }
  }

  void ComputeFluxJacobianDotN(const mfem::Vector&,
                               const mfem::Vector& normal,
                               mfem::ElementTransformation&,
                               mfem::DenseMatrix& jacobian) const override
  {
    jacobian.SetSize(num_equations, num_equations);
    double normal_scale = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      normal_scale += dir_(d) * normal(d);
    }
    for (int equation = 0; equation < num_equations; ++equation)
    {
      for (int state_component = 0; state_component < num_equations;
           ++state_component)
      {
        jacobian(equation, state_component) =
          equation == state_component ? normal_scale : 0.0;
      }
    }
  }

private:
  mfem::Vector dir_;
};

double
DotMomentumNormal(const mfem::Vector& cons, const mfem::Vector& normal, int dim)
{
  double un = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    un += cons(1 + d) * normal(d);
  }
  return un;
}

double ExactAvgUnOverRho(const mfem::Vector& state1,
                         const mfem::Vector& state2,
                         const mfem::Vector& normal,
                         int dim)
{
  const double rho1 = state1(0);
  const double rho2 = state2(0);
  MFEM_VERIFY(rho1 > 0.0 && rho2 > 0.0,
              "ExactAvgUnOverRho requires positive density endpoints.");
  const double drho = rho2 - rho1;

  double mn1 = 0.0;
  double mn2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    mn1 += state1(1 + d) * normal(d);
    mn2 += state2(1 + d) * normal(d);
  }
  const double dmn = mn2 - mn1;

  const double eps = 1.0e-14 * std::max(1.0, std::abs(rho1));
  if (std::abs(drho) <= eps)
  {
    return (mn1 + 0.5 * dmn) / rho1;
  }

  const double inv_drho = 1.0 / drho;
  const double inv_drho2 = inv_drho * inv_drho;
  const double log_ratio = std::log1p(drho / rho1);
  return dmn * inv_drho + (mn1 * drho - dmn * rho1) * inv_drho2 * log_ratio;
}

void AddSigmaDotNTerms(const mfem::Vector& state,
                       const mfem::Vector& normal,
                       double sigma,
                       mfem::Vector& fluxN)
{
  const int dim = normal.Size();
  const double rho_floor = mfem::igrlab::FluxDefaults::kRhoVelFloor;
  const double rho = std::max(state(0), rho_floor);

  for (int d = 0; d < dim; ++d)
  {
    fluxN(1 + d) += sigma * normal(d);
  }

  const double un = DotMomentumNormal(state, normal, dim) / rho;
  fluxN(1 + dim) += sigma * un;
}

double SigmaAtElementPoint(const mfem::ParGridFunction& sigma,
                           mfem::ElementTransformation& T,
                           const mfem::IntegrationPoint& ip)
{
  T.SetIntPoint(&ip);
  return sigma.GetValue(T, ip);
}

struct SPDProbeSummary
{
  double max_sym_rel = 0.0;
  double min_rayleigh = std::numeric_limits<double>::infinity();
  bool finite = true;
};

double GlobalDotMPI(const mfem::Vector& x, const mfem::Vector& y, MPI_Comm comm)
{
  const double local = x * y;
  double global = 0.0;
  MPI_Allreduce(&local,
                &global,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM,
                comm);
  return global;
}

void FillDeterministicProbeVector(mfem::Vector& v, int seed)
{
  const int rank = mfem::Mpi::WorldRank();
  for (int i = 0; i < v.Size(); ++i)
  {
    const double t = static_cast<double>((i + 1) * (seed + 5) + 19 * rank);
    v(i) = std::sin(0.017 * t) + 0.5 * std::cos(0.031 * t);
  }
}

bool SameVectorBits(const mfem::Vector& a, const mfem::Vector& b)
{
  if (a.Size() != b.Size())
  {
    return false;
  }
  if (a.Size() == 0)
  {
    return true;
  }
  const mfem::real_t* a_data = a.HostRead();
  const mfem::real_t* b_data = b.HostRead();
  return std::memcmp(a_data,
                     b_data,
                     static_cast<std::size_t>(a.Size()) *
                       sizeof(mfem::real_t)) == 0;
}

SPDProbeSummary
ProbeSPD(const mfem::HypreParMatrix& A, MPI_Comm comm, int samples = 4)
{
  SPDProbeSummary out;
  const int n = A.Width();
  MFEM_VERIFY(n > 0, "SPD probe requires non-empty operator.");

  mfem::Vector x(n), y(n), Ax(n), Ay(n);
  for (int s = 0; s < samples; ++s)
  {
    FillDeterministicProbeVector(x, 2 * s + 1);
    FillDeterministicProbeVector(y, 2 * s + 2);
    A.Mult(x, Ax);
    A.Mult(y, Ay);

    for (int i = 0; i < n; ++i)
    {
      if (!std::isfinite(Ax(i)) || !std::isfinite(Ay(i)))
      {
        out.finite = false;
      }
    }

    const double xAy = GlobalDotMPI(x, Ay, comm);
    const double yAx = GlobalDotMPI(y, Ax, comm);
    const double sym_rel =
      std::abs(xAy - yAx) / std::max({1.0, std::abs(xAy), std::abs(yAx)});
    out.max_sym_rel = std::max(out.max_sym_rel, sym_rel);

    const double xAx = GlobalDotMPI(x, Ax, comm);
    const double x2 = GlobalDotMPI(x, x, comm);
    const double rayleigh = xAx / std::max(x2, 1.0e-30);
    out.min_rayleigh = std::min(out.min_rayleigh, rayleigh);
  }

  return out;
}

class ScopedEnvVar
{
public:
  ScopedEnvVar(const char* name, const char* value)
    : name_(name)
  {
    const char* prev = igrlab::GetEnv(name_.c_str());
    if (prev)
    {
      had_prev_ = true;
      prev_value_ = prev;
    }
    setenv(name_.c_str(), value, 1);
  }

  ~ScopedEnvVar()
  {
    if (had_prev_)
    {
      setenv(name_.c_str(), prev_value_.c_str(), 1);
    }
    else
    {
      unsetenv(name_.c_str());
    }
  }

private:
  std::string name_;
  bool had_prev_ = false;
  std::string prev_value_;
};

} // namespace

int RunIGRTests(TestContext& ctx)
{
  constexpr int dim = 2;
  constexpr double gamma = 1.4;
  const double tol = 1.0e-11;

  // --- Manufactured primitive-gradient source and quotient invariance ---
  {
    mfem::Mesh source_mesh = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    mfem::ParMesh source_pmesh(MPI_COMM_SELF, source_mesh);
    mfem::DG_FECollection source_fec(1,
                                     dim,
                                     mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace source_fes(
      &source_pmesh, &source_fec, dim + 2, mfem::Ordering::byNODES);
    mfem::ParGridFunction source_state(&source_fes);
    constexpr double source_alpha = 0.37;
    BrokenDGIGRSourceCoefficient source(
      source_state, dim, dim + 2, source_alpha, 5.0e-7);

    auto SetManufacturedPoint = [&](double rho,
                                    const mfem::Vector& velocity,
                                    const mfem::Vector& grad_rho,
                                    const mfem::DenseMatrix& grad_velocity)
    {
      mfem::Vector& state_scratch = source.StateScratch();
      mfem::DenseMatrix& gradient_scratch = source.GradientScratch();
      state_scratch = 0.0;
      gradient_scratch = 0.0;
      state_scratch(0) = rho;
      state_scratch(1 + dim) = 1.0;
      for (int j = 0; j < dim; ++j)
      {
        gradient_scratch(0, j) = grad_rho(j);
      }
      for (int i = 0; i < dim; ++i)
      {
        state_scratch(1 + i) = rho * velocity(i);
        for (int j = 0; j < dim; ++j)
        {
          gradient_scratch(1 + i, j) =
            velocity(i) * grad_rho(j) + rho * grad_velocity(i, j);
        }
      }
    };

    mfem::Vector velocity({2.1, -0.7});
    mfem::Vector grad_rho({0.8, -1.3});
    mfem::DenseMatrix strain(dim);
    strain(0, 0) = 1.2;
    strain(0, 1) = -0.4;
    strain(1, 0) = 0.7;
    strain(1, 1) = -0.3;
    const double strain_div = strain(0, 0) + strain(1, 1);
    const double strain_tr_a2 =
      strain(0, 0) * strain(0, 0) +
      2.0 * strain(0, 1) * strain(1, 0) +
      strain(1, 1) * strain(1, 1);
    const double expected_strain_source =
      source_alpha * (strain_tr_a2 + strain_div * strain_div);

    SetManufacturedPoint(1.7, velocity, grad_rho, strain);
    const double variable_density_source = source.EvalFromStateAndGradient();
    SetManufacturedPoint(0.002, velocity, grad_rho, strain);
    const double low_density_source = source.EvalFromStateAndGradient();
    ctx.CheckNear(variable_density_source,
                  expected_strain_source,
                  2.0e-12,
                  "IGR source quotient recovers manufactured velocity gradient");
    ctx.CheckNear(low_density_source,
                  expected_strain_source,
                  2.0e-12,
                  "IGR source has no spurious density scaling above the floor");

    mfem::DenseMatrix rotation(dim);
    rotation = 0.0;
    rotation(0, 1) = -2.0;
    rotation(1, 0) = 2.0;
    grad_rho = 0.0;
    SetManufacturedPoint(1.0, velocity, grad_rho, rotation);
    ctx.CheckNear(source.EvalFromStateAndGradient(),
                  -8.0 * source_alpha,
                  2.0e-12,
                  "IGR multidimensional source retains its rotational sign");
  }

  mfem::EulerFlux base_flux(dim, gamma);
  mfem::FluxWithIGR igr_flux(base_flux, gamma);

  const mfem::Vector state = EulerConsFromPrim2D(
    /*rho=*/1.2,
    /*u=*/2.0,
    /*v=*/-0.5,
    /*p=*/1.0,
    gamma);
  mfem::Vector normal(dim);
  normal(0) = 2.0;
  normal(1) = -1.0;

  // --- With sigma disabled, the decorated flux matches the base flux ---
  {
    mfem::FaceElementTransformations Tr_dummy;
    mfem::Vector base_fn(dim + 2);
    mfem::Vector igr_fn(dim + 2);
    const double base_speed =
      base_flux.ComputeFluxDotN(state, normal, Tr_dummy, base_fn);
    const double igr_speed =
      igr_flux.ComputeFluxDotN(state, normal, Tr_dummy, igr_fn);
    ctx.CheckNear(igr_speed,
                  base_speed,
                  tol,
                  "FluxWithIGR preserves Euler speed when sigma is null");
    ctx.AddParity(igr_fn(0) + igr_fn(1) + igr_fn(2) + igr_fn(3));
    for (int i = 0; i < base_fn.Size(); ++i)
    {
      ctx.CheckNear(
        igr_fn(i),
        base_fn(i),
        tol,
        "FluxWithIGR matches base flux when sigma is null, component " +
          std::to_string(i));
    }
  }

  // --- With constant sigma, momentum/energy offsets match the documented form ---
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    /*nx=*/2,
    /*ny=*/2,
    mfem::Element::QUADRILATERAL,
    /*generate_edges=*/true,
    /*sx=*/1.0,
    /*sy=*/1.0);
  mfem::ParMesh pmesh(MPI_COMM_SELF, mesh);
  MFEM_VERIFY(pmesh.GetNumFaces() > 0,
              "IGR flux tests require a mesh with faces.");
  mfem::DG_FECollection fec(/*order=*/0, dim);
  mfem::ParFiniteElementSpace sigma_fes(&pmesh, &fec, /*vdim=*/1);
  mfem::ParGridFunction sigma(&sigma_fes);
  const double sigma_value = 0.25;
  mfem::ConstantCoefficient sigma_coeff(sigma_value);
  sigma.ProjectCoefficient(sigma_coeff);
  sigma.ExchangeFaceNbrData();

  igr_flux.SetSigmaField(&sigma);
  ctx.Check(!igr_flux.GetSigmaAwareCFL(),
            "FluxWithIGR defaults to Euler-only CFL accounting");
  ctx.Check(!igr_flux.GetSigmaAwareRusanov(),
            "FluxWithIGR defaults to Euler-only Rusanov dissipation");
  ctx.Check(!igr_flux.GetInvariantDomainFlux(),
            "FluxWithIGR defaults to the legacy Euler-only speed policy");
  ctx.CheckNear(igr_flux.GetGamma(),
                gamma,
                0.0,
                "FluxWithIGR retains the supplied Euler gamma");

  mfem::FaceElementTransformations& Tr_face = GetFaceTr(pmesh, /*face=*/0);

  mfem::Vector base_fn(dim + 2);
  mfem::Vector igr_fn(dim + 2);
  const double base_speed =
    base_flux.ComputeFluxDotN(state, normal, Tr_face, base_fn);
  const double default_igr_speed =
    igr_flux.ComputeFluxDotN(state, normal, Tr_face, igr_fn);
  ctx.AddParity(igr_fn(0) + igr_fn(1) + igr_fn(2) + igr_fn(3));

  const double rho = state(0);
  const double un = DotMomentumNormal(state, normal, dim) / rho;

  // Momentum rows get +sigma*n, energy row gets +sigma*(u·n).
  ctx.CheckNear(igr_fn(1) - base_fn(1),
                sigma_value * normal(0),
                tol,
                "FluxWithIGR adds sigma*n_x to momentum_x");
  ctx.CheckNear(igr_fn(2) - base_fn(2),
                sigma_value * normal(1),
                tol,
                "FluxWithIGR adds sigma*n_y to momentum_y");
  ctx.CheckNear(igr_fn(3) - base_fn(3),
                sigma_value * un,
                tol,
                "FluxWithIGR adds sigma*(u·n) to energy flux");
  ctx.CheckNear(default_igr_speed,
                base_speed,
                tol,
                "FluxWithIGR defaults to the Euler face speed");

  igr_flux.SetSigmaAwareCFL(true);
  mfem::Vector sigma_cfl_fn(dim + 2);
  const double igr_speed =
    igr_flux.ComputeFluxDotN(state, normal, Tr_face, sigma_cfl_fn);
  for (int i = 0; i < igr_fn.Size(); ++i)
  {
    ctx.CheckNear(sigma_cfl_fn(i),
                  igr_fn(i),
                  tol,
                  "Sigma-aware CFL does not alter physical flux, component " +
                    std::to_string(i));
  }
  ctx.CheckNear(igr_speed,
                base_speed + std::sqrt(std::abs(sigma_value) / rho),
                tol,
                "FluxWithIGR opt-in includes sigma pressure scale in face speed");

  // The typed extrapolated-DGSEM mode uses the exact local frozen-sigma
  // acoustic spectrum for both CFL reporting and Rusanov dissipation.
  igr_flux.SetSigmaAwareCFL(false);
  igr_flux.SetSigmaAwareRusanov(false);
  mfem::FluxWithIGR frozen_spectral_igr_flux(base_flux, gamma);
  frozen_spectral_igr_flux.SetSigmaField(&sigma);
  frozen_spectral_igr_flux.SetFrozenSigmaSpectralSpeed(true);
  mfem::Vector frozen_spectral_flux(dim + 2);
  const double frozen_spectral_speed =
    frozen_spectral_igr_flux.ComputeFluxDotN(
      state, normal, Tr_face, frozen_spectral_flux);
  const double normal_norm = normal.Norml2();
  const double unit_normal_velocity = std::abs(un) / normal_norm;
  const double pressure = 1.0;
  const double q_sigma = gamma * pressure + (gamma - 1.0) * sigma_value;
  const double expected_frozen_spectral_speed =
    unit_normal_velocity + std::sqrt(q_sigma / rho);
  ctx.Check(frozen_spectral_igr_flux.GetFrozenSigmaSpectralSpeed(),
            "FluxWithIGR exposes the typed frozen-sigma spectral policy");
  ctx.CheckNear(frozen_spectral_speed,
                expected_frozen_spectral_speed,
                tol,
                "FluxWithIGR reports the exact frozen-sigma normal spectrum");
  ctx.CheckNear(
    frozen_spectral_igr_flux.SelectRusanovDissipationSpeed(
      frozen_spectral_speed, rho, sigma_value),
    expected_frozen_spectral_speed,
    0.0,
    "FluxWithIGR couples frozen-sigma Rusanov dissipation to its CFL speed");
  for (int i = 0; i < igr_fn.Size(); ++i)
  {
    ctx.CheckNear(frozen_spectral_flux(i),
                  igr_fn(i),
                  tol,
                  "Frozen-sigma spectral mode preserves physical flux, component " +
                    std::to_string(i));
  }
  // The IGR bound is the exact pressure threshold for both frozen-stage
  // Lax-Friedrichs split states. This low-pressure case distinguishes it from
  // the former sqrt(|sigma|/rho) heuristic by several orders of magnitude.
  {
    const mfem::Vector low_pressure_state = EulerConsFromPrim2D(
      /*rho=*/1.0e-2,
      /*u=*/0.2,
      /*v=*/0.3,
      /*p=*/1.0e-6,
      gamma);
    mfem::Vector unit_normal(dim);
    unit_normal(0) = 1.0;
    unit_normal(1) = 0.0;
    mfem::Vector full_flux(dim + 2);
    const double current_speed = igr_flux.ComputeFluxDotN(
      low_pressure_state, unit_normal, Tr_face, full_flux);
    const double analytic_threshold =
      0.2 + std::abs(1.0e-6 + sigma_value) *
              std::sqrt((gamma - 1.0) /
                        (2.0 * 1.0e-2 * 1.0e-6));
    const double split_threshold = mfem::IGRFluxInvariantDomainSpeed(
      current_speed,
      1.0e-2,
      1.0e-6,
      0.2,
      sigma_value,
      gamma,
      igr_flux.GetVelocityRhoFloor());
    ctx.CheckNear(split_threshold,
                  analytic_threshold,
                  2.0e-10 * analytic_threshold,
                  "IGR invariant-domain speed matches the analytic threshold");

    auto pressure_of = [gamma, dim](const mfem::Vector& value)
    {
      double momentum_sq = 0.0;
      for (int d = 0; d < dim; ++d)
      {
        momentum_sq += value(1 + d) * value(1 + d);
      }
      return (gamma - 1.0) *
             (value(1 + dim) - 0.5 * momentum_sq / value(0));
    };
    bool safe_splits = true;
    bool unsafe_split_detected = false;
    for (const double sign : {-1.0, 1.0})
    {
      mfem::Vector safe(low_pressure_state);
      safe.Add(sign / (1.001 * split_threshold), full_flux);
      safe_splits = safe_splits && safe(0) > 0.0 && pressure_of(safe) > 0.0;

      mfem::Vector unsafe(low_pressure_state);
      unsafe.Add(sign / (0.999 * split_threshold), full_flux);
      unsafe_split_detected = unsafe_split_detected ||
                              unsafe(0) <= 0.0 || pressure_of(unsafe) <= 0.0;
    }
    ctx.Check(safe_splits,
              "IGR invariant-domain speed keeps both split states admissible");
    ctx.Check(unsafe_split_detected,
              "A speed below the IGR invariant-domain threshold loses pressure");

    igr_flux.SetSigmaAwareCFL(false);
    igr_flux.SetSigmaAwareRusanov(false);
    igr_flux.SetInvariantDomainFlux(true);
    mfem::Vector exact_flux(dim + 2);
    const double exact_reported_speed = igr_flux.ComputeFluxDotN(
      low_pressure_state, unit_normal, Tr_face, exact_flux);
    const double exact_dissipation_speed =
      igr_flux.SelectRusanovDissipationSpeed(exact_reported_speed,
                                             low_pressure_state(0),
                                             sigma_value);
    ctx.CheckNear(exact_reported_speed,
                  analytic_threshold,
                  2.0e-10 * analytic_threshold,
                  "Exact IGR mode reports the invariant-domain CFL speed");
    ctx.CheckNear(exact_dissipation_speed,
                  exact_reported_speed,
                  0.0,
                  "Exact IGR mode couples Rusanov dissipation to CFL speed");
    for (int i = 0; i < exact_flux.Size(); ++i)
    {
      ctx.CheckNear(exact_flux(i),
                    full_flux(i),
                    tol,
                    "Exact IGR speed mode preserves physical flux, component " +
                      std::to_string(i));
    }

    igr_flux.SetSigmaAwareCFL(true);
    igr_flux.SetSigmaAwareRusanov(true);
    mfem::Vector exact_with_heuristics_flux(dim + 2);
    const double exact_with_heuristics = igr_flux.ComputeFluxDotN(
      low_pressure_state, unit_normal, Tr_face, exact_with_heuristics_flux);
    ctx.CheckNear(exact_with_heuristics,
                  exact_reported_speed,
                  0.0,
                  "Exact IGR mode overrides independent sigma heuristics");
    const mfem::IGRFrozenSigmaTelemetry q_sigma_telemetry =
      igr_flux.GetFrozenSigmaTelemetry();
    const double expected_q_sigma =
      gamma * 1.0e-6 + (gamma - 1.0) * sigma_value;
    ctx.Check(q_sigma_telemetry.enabled &&
                q_sigma_telemetry.sample_count == 2 &&
                q_sigma_telemetry.nonpositive_count == 0 &&
                q_sigma_telemetry.nonfinite_count == 0,
              "Exact IGR host speed telemetry counts attempted samples");
    ctx.CheckNear(q_sigma_telemetry.minimum_raw_q_sigma,
                  expected_q_sigma,
                  2.0e-14,
                  "Exact IGR host telemetry records minimum raw q_sigma");
    ctx.CheckNear(q_sigma_telemetry.minimum_q_sigma_over_gamma_p,
                  expected_q_sigma / (gamma * 1.0e-6),
                  2.0e-14,
                  "Exact IGR host telemetry records normalized q_sigma");
    igr_flux.SetInvariantDomainFlux(false);
  }

  // Exercise the raw invariant-domain algebra away from the default gas and
  // with a negative Sigma.  The state is only one representable value above
  // its configured density and pressure floors; scaled oblique normals must
  // change F.n linearly without changing the reported unit-normal speed.
  {
    constexpr double gamma_floor = 1.67;
    constexpr double rho_floor = 5.0e-7;
    constexpr double pressure_floor = 1.0e-6;
    constexpr double negative_sigma = -9.0e-6;
    const double rho_floor_state =
      std::nextafter(rho_floor, std::numeric_limits<double>::infinity());
    const double pressure_floor_state =
      std::nextafter(pressure_floor,
                     std::numeric_limits<double>::infinity());
    const mfem::Vector floor_state = EulerConsFromPrim2D(
      rho_floor_state,
      /*u=*/0.4,
      /*v=*/-0.3,
      pressure_floor_state,
      gamma_floor);
    mfem::Vector unit_oblique({0.6, -0.8});
    constexpr double normal_scale = 7.25;
    mfem::Vector scaled_oblique(unit_oblique);
    scaled_oblique *= normal_scale;

    mfem::EulerFlux floor_base_flux(dim, gamma_floor);
    mfem::FaceElementTransformations dummy_face;
    mfem::Vector unit_flux(dim + 2);
    mfem::Vector scaled_flux(dim + 2);
    const double unit_base_speed = floor_base_flux.ComputeFluxDotN(
      floor_state, unit_oblique, dummy_face, unit_flux);
    const double scaled_base_speed = floor_base_flux.ComputeFluxDotN(
      floor_state, scaled_oblique, dummy_face, scaled_flux);
    ::igrlab::frozen_scalar_igr_algebra::AddNormalSigmaFlux(
      dim,
      floor_state.GetData(),
      1,
      unit_oblique.GetData(),
      1,
      negative_sigma,
      rho_floor,
      unit_flux.GetData(),
      1);
    ::igrlab::frozen_scalar_igr_algebra::AddNormalSigmaFlux(
      dim,
      floor_state.GetData(),
      1,
      scaled_oblique.GetData(),
      1,
      negative_sigma,
      rho_floor,
      scaled_flux.GetData(),
      1);

    const double transport_speed = std::abs(0.4 * unit_oblique(0) -
                                            0.3 * unit_oblique(1));
    const double expected_split_speed =
      transport_speed +
      std::abs(pressure_floor_state + negative_sigma) *
        std::sqrt((gamma_floor - 1.0) /
                  (2.0 * rho_floor_state * pressure_floor_state));
    const double invariant_speed = mfem::IGRFluxInvariantDomainSpeed(
      unit_base_speed,
      rho_floor_state,
      pressure_floor_state,
      transport_speed,
      negative_sigma,
      gamma_floor,
      rho_floor,
      pressure_floor);
    ctx.Check(expected_split_speed > unit_base_speed,
              "Near-floor negative-Sigma case exercises the split threshold");
    ctx.CheckNear(invariant_speed,
                  expected_split_speed,
                  2.0e-13 * expected_split_speed,
                  "Invariant-domain algebra handles negative Sigma, nondefault "
                  "gamma, and configured floors");
    ctx.CheckNear(scaled_base_speed,
                  unit_base_speed,
                  2.0e-13 * unit_base_speed,
                  "Euler unit-normal speed is invariant to normal scaling");
    for (int eq = 0; eq < unit_flux.Size(); ++eq)
    {
      ctx.CheckNear(scaled_flux(eq),
                    normal_scale * unit_flux(eq),
                    2.0e-12 * std::max(1.0, std::abs(scaled_flux(eq))),
                    "Euler+IGR normal flux scales linearly for an oblique "
                    "normal, component " +
                      std::to_string(eq));
    }

    const auto pressure_of = [&](const mfem::Vector& value)
    {
      const double momentum_squared =
        value(1) * value(1) + value(2) * value(2);
      return (gamma_floor - 1.0) *
             (value(3) - 0.5 * momentum_squared / value(0));
    };
    bool safe_splits = true;
    bool unsafe_split = false;
    for (const double sign : {-1.0, 1.0})
    {
      mfem::Vector safe(floor_state);
      safe.Add(sign / (1.01 * invariant_speed), unit_flux);
      safe_splits = safe_splits && safe(0) > 0.0 && pressure_of(safe) > 0.0;

      mfem::Vector unsafe(floor_state);
      unsafe.Add(sign / (0.99 * invariant_speed), unit_flux);
      unsafe_split = unsafe_split ||
                     unsafe(0) <= 0.0 || pressure_of(unsafe) <= 0.0;
    }
    ctx.Check(safe_splits,
              "Negative-Sigma invariant speed keeps both near-floor splits "
              "admissible");
    ctx.Check(unsafe_split,
              "Negative-Sigma split threshold is sharp in the near-floor "
              "case");
  }

  // --- Tensor flux path: sigma terms must enter ComputeFlux (volume flux) ---
  {
    mfem::ElementTransformation* Te = pmesh.GetElementTransformation(0);
    MFEM_VERIFY(Te != nullptr, "Failed to obtain element transformation.");
    const mfem::IntegrationRule& ir_elem =
      mfem::IntRules.Get(Te->GetGeometryType(), /*order=*/1);
    const mfem::IntegrationPoint& ip_elem = ir_elem.IntPoint(0);
    Te->SetIntPoint(&ip_elem);

    mfem::DenseMatrix base_F(dim + 2, dim);
    mfem::DenseMatrix igr_F(dim + 2, dim);
    base_flux.ComputeFlux(state, *Te, base_F);
    igr_flux.ComputeFlux(state, *Te, igr_F);

    const double rho_vol = state(0);
    for (int d = 0; d < dim; ++d)
    {
      ctx.CheckNear(
        igr_F(1 + d, d) - base_F(1 + d, d),
        sigma_value,
        tol,
        "FluxWithIGR adds +sigma to momentum tensor diagonal, dim=" +
          std::to_string(d));
      const double expected_energy = sigma_value * state(1 + d) / rho_vol;
      ctx.CheckNear(igr_F(1 + dim, d) - base_F(1 + dim, d),
                    expected_energy,
                    tol,
                    "FluxWithIGR adds +sigma*u to energy tensor row, dim=" +
                      std::to_string(d));
    }

    mfem::Vector jacobian_direction(dim);
    jacobian_direction = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      jacobian_direction(d) = 0.25 * (d + 1);
    }
    LinearAverageFlux jacobian_base(dim, jacobian_direction);
    mfem::FluxWithIGR jacobian_igr_flux(jacobian_base);
    jacobian_igr_flux.SetSigmaField(&sigma);
    mfem::DenseTensor base_tensor_jacobian(dim + 2, dim + 2, dim);
    mfem::DenseTensor igr_tensor_jacobian(dim + 2, dim + 2, dim);
    jacobian_base.ComputeFluxJacobian(state, *Te, base_tensor_jacobian);
    jacobian_igr_flux.ComputeFluxJacobian(state, *Te, igr_tensor_jacobian);
    mfem::DenseMatrix base_normal_jacobian(dim + 2, dim + 2);
    mfem::DenseMatrix igr_normal_jacobian(dim + 2, dim + 2);
    jacobian_base.ComputeFluxJacobianDotN(state,
                                          normal,
                                          *Te,
                                          base_normal_jacobian);
    jacobian_igr_flux.ComputeFluxJacobianDotN(state,
                                               normal,
                                               *Te,
                                               igr_normal_jacobian);
    for (int equation = 0; equation < dim + 2; ++equation)
    {
      for (int state_component = 0; state_component < dim + 2;
           ++state_component)
      {
        double expected_normal = 0.0;
        if (equation == dim + 1 && state_component == 0)
        {
          expected_normal =
            -sigma_value * DotMomentumNormal(state, normal, dim) / (rho * rho);
        }
        if (equation == dim + 1 && state_component >= 1 &&
            state_component <= dim)
        {
          expected_normal = sigma_value * normal(state_component - 1) / rho;
        }
        ctx.CheckNear(igr_normal_jacobian(equation, state_component) -
                        base_normal_jacobian(equation, state_component),
                      expected_normal,
                      tol,
                      "FluxWithIGR adds frozen-sigma normal Jacobian dim=" +
                        std::to_string(state_component));
        for (int d = 0; d < dim; ++d)
        {
          double expected_tensor = 0.0;
          if (equation == dim + 1 && state_component == 0)
          {
            expected_tensor = -sigma_value * state(1 + d) / (rho * rho);
          }
          if (equation == dim + 1 && state_component == 1 + d)
          {
            expected_tensor = sigma_value / rho;
          }
          ctx.CheckNear(igr_tensor_jacobian(equation, state_component, d) -
                          base_tensor_jacobian(equation, state_component, d),
                        expected_tensor,
                        tol,
                        "FluxWithIGR adds frozen-sigma tensor Jacobian dim=" +
                          std::to_string(d));
        }
      }
    }

    jacobian_igr_flux.SetSigmaField(nullptr);
    mfem::DenseTensor null_sigma_tensor_jacobian(dim + 2, dim + 2, dim);
    mfem::DenseMatrix null_sigma_normal_jacobian(dim + 2, dim + 2);
    jacobian_igr_flux.ComputeFluxJacobian(state,
                                           *Te,
                                           null_sigma_tensor_jacobian);
    jacobian_igr_flux.ComputeFluxJacobianDotN(state,
                                               normal,
                                               *Te,
                                               null_sigma_normal_jacobian);
    for (int equation = 0; equation < dim + 2; ++equation)
    {
      for (int state_component = 0; state_component < dim + 2;
           ++state_component)
      {
        ctx.CheckNear(null_sigma_normal_jacobian(equation, state_component),
                      base_normal_jacobian(equation, state_component),
                      tol,
                      "FluxWithIGR null sigma normal Jacobian passthrough");
        for (int d = 0; d < dim; ++d)
        {
          ctx.CheckNear(null_sigma_tensor_jacobian(equation,
                                                    state_component,
                                                    d),
                        base_tensor_jacobian(equation, state_component, d),
                        tol,
                        "FluxWithIGR null sigma tensor Jacobian passthrough");
        }
      }
    }
  }

  // --- Exact averaged sigma correction for side-explicit average flux path ---
  {
    mfem::Vector dir(dim);
    dir(0) = 1.0;
    dir(1) = -0.25;
    LinearAverageFlux linear_base(dim, dir);
    mfem::FluxWithIGR igr_avg(linear_base);
    igr_avg.SetSigmaField(&sigma);

    const mfem::Vector state_avg_1 = EulerConsFromPrim2D(
      /*rho=*/0.9,
      /*u=*/1.8,
      /*v=*/-0.3,
      /*p=*/1.0,
      gamma);
    const mfem::Vector state_avg_2 = EulerConsFromPrim2D(
      /*rho=*/1.4,
      /*u=*/-0.2,
      /*v=*/0.6,
      /*p=*/0.9,
      gamma);

    mfem::Vector base_avg_fn(dim + 2);
    mfem::Vector igr_avg_fn(dim + 2);
    linear_base.ComputeAvgFluxDotN(state_avg_1,
                                   state_avg_2,
                                   normal,
                                   Tr_face,
                                   base_avg_fn);
    igr_avg.ComputeAvgFluxDotNAtSide(state_avg_1,
                                     state_avg_2,
                                     normal,
                                     Tr_face,
                                     mfem::FaceSide::Left,
                                     igr_avg_fn);

    mfem::Vector expected_avg(base_avg_fn);
    for (int d = 0; d < dim; ++d)
    {
      expected_avg(1 + d) += sigma_value * normal(d);
    }
    expected_avg(1 + dim) +=
      sigma_value * ExactAvgUnOverRho(state_avg_1, state_avg_2, normal, dim);

    for (int i = 0; i < expected_avg.Size(); ++i)
    {
      ctx.CheckNear(igr_avg_fn(i),
                    expected_avg(i),
                    5.0e-12,
                    "FluxWithIGR exact averaged sigma correction, component " +
                      std::to_string(i));
    }
  }

  // --- Interior-face side selection: sigma should come from the matching side ---
  {
    mfem::Mesh mesh_if = mfem::Mesh::MakeCartesian2D(
      /*nx=*/4,
      /*ny=*/2,
      mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    mfem::ParMesh pmesh_if(MPI_COMM_SELF, mesh_if);

    mfem::DG_FECollection fec_if(/*order=*/0, dim);
    mfem::ParFiniteElementSpace sigma_fes_if(&pmesh_if, &fec_if, /*vdim=*/1);
    mfem::ParGridFunction sigma_if(&sigma_fes_if);
    const double sigma_left = 0.1;
    const double sigma_right = 0.35;
    StepSigmaCoefficient sigma_step(sigma_left, sigma_right, /*split_x=*/0.5);
    sigma_if.ProjectCoefficient(sigma_step);
    sigma_if.ExchangeFaceNbrData();

    mfem::FluxWithIGR igr_flux_if(base_flux);
    igr_flux_if.SetSigmaField(&sigma_if);
    igr_flux_if.SetSigmaAwareCFL(true);
    igr_flux_if.SetSigmaAwareRusanov(true);
    mfem::RusanovFluxWithSides num_flux(igr_flux_if);

    const mfem::Vector state_left = EulerConsFromPrim2D(
      /*rho=*/1.0,
      /*u=*/1.5,
      /*v=*/0.25,
      /*p=*/1.0,
      gamma);
    const mfem::Vector state_right = EulerConsFromPrim2D(
      /*rho=*/0.8,
      /*u=*/-0.75,
      /*v=*/0.5,
      /*p=*/0.9,
      gamma);
    mfem::Vector normal_if(dim);
    normal_if(0) = 1.25;
    normal_if(1) = -0.5;

    double local_max_err = 0.0;
    double local_base_dissipation_err = 0.0;
    double local_cfl_separation_err = 0.0;
    int local_face_count = 0;

    const mfem::IntegrationRule& ir_face =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, /*order=*/1);
    const mfem::IntegrationPoint& ip_face = ir_face.IntPoint(0);

    for (int f = 0; f < pmesh_if.GetNumFaces(); ++f)
    {
      mfem::FaceElementTransformations* Tr =
        pmesh_if.GetFaceElementTransformations(f);
      if (!Tr || !Tr->Elem2)
      {
        continue;
      }
      Tr->SetAllIntPoints(&ip_face);

      const mfem::IntegrationPoint& ip1 = Tr->GetElement1IntPoint();
      const mfem::IntegrationPoint& ip2 = Tr->GetElement2IntPoint();
      mfem::ElementTransformation& T1 = *Tr->Elem1;
      mfem::ElementTransformation& T2 = *Tr->Elem2;
      const double sigma1 = SigmaAtElementPoint(sigma_if, T1, ip1);
      const double sigma2 = SigmaAtElementPoint(sigma_if, T2, ip2);

      mfem::Vector fluxN1(dim + 2);
      mfem::Vector fluxN2(dim + 2);
      const mfem::real_t speed1 =
        base_flux.ComputeFluxDotN(state_left, normal_if, *Tr, fluxN1);
      const mfem::real_t speed2 =
        base_flux.ComputeFluxDotN(state_right, normal_if, *Tr, fluxN2);

      AddSigmaDotNTerms(state_left, normal_if, sigma1, fluxN1);
      AddSigmaDotNTerms(state_right, normal_if, sigma2, fluxN2);

      const double igr_speed1 = mfem::IGRFluxCharacteristicSpeed(
        speed1, state_left(0), sigma1, igr_flux_if.GetVelocityRhoFloor());
      const double igr_speed2 = mfem::IGRFluxCharacteristicSpeed(
        speed2, state_right(0), sigma2, igr_flux_if.GetVelocityRhoFloor());
      const double max_speed = std::max(igr_speed1, igr_speed2);
      const double scaled_max_speed = max_speed * normal_if.Norml2();

      mfem::Vector expected(dim + 2);
      for (int i = 0; i < expected.Size(); ++i)
      {
        expected(i) =
          0.5 * (scaled_max_speed * (state_left(i) - state_right(i)) +
                 fluxN1(i) + fluxN2(i));
      }

      mfem::Vector actual(dim + 2);
      const double actual_cfl_speed =
        num_flux.Eval(state_left, state_right, normal_if, *Tr, actual);

      for (int i = 0; i < actual.Size(); ++i)
      {
        local_max_err =
          std::max(local_max_err, std::abs(actual(i) - expected(i)));
      }

      igr_flux_if.SetSigmaAwareRusanov(false);
      mfem::Vector base_dissipation_actual(dim + 2);
      const double base_dissipation_cfl = num_flux.Eval(
        state_left, state_right, normal_if, *Tr, base_dissipation_actual);
      const double base_max_speed = std::max(speed1, speed2);
      const double base_scaled_max_speed = base_max_speed * normal_if.Norml2();
      for (int i = 0; i < base_dissipation_actual.Size(); ++i)
      {
        const double base_expected =
          0.5 * (base_scaled_max_speed *
                   (state_left(i) - state_right(i)) +
                 fluxN1(i) + fluxN2(i));
        local_base_dissipation_err =
          std::max(local_base_dissipation_err,
                   std::abs(base_dissipation_actual(i) - base_expected));
      }
      local_cfl_separation_err =
        std::max(local_cfl_separation_err,
                 std::abs(base_dissipation_cfl - max_speed));

      igr_flux_if.SetSigmaAwareRusanov(true);
      igr_flux_if.SetSigmaAwareCFL(false);
      mfem::Vector base_cfl_actual(dim + 2);
      const double base_cfl_speed = num_flux.Eval(
        state_left, state_right, normal_if, *Tr, base_cfl_actual);
      for (int i = 0; i < base_cfl_actual.Size(); ++i)
      {
        local_cfl_separation_err =
          std::max(local_cfl_separation_err,
                   std::abs(base_cfl_actual(i) - actual(i)));
      }
      local_cfl_separation_err =
        std::max(local_cfl_separation_err,
                 std::abs(base_cfl_speed - base_max_speed));
      local_cfl_separation_err =
        std::max(local_cfl_separation_err,
                 std::abs(actual_cfl_speed - max_speed));
      igr_flux_if.SetSigmaAwareCFL(true);
      ++local_face_count;
    }

    const double global_faces =
      GlobalSum(static_cast<double>(local_face_count));
    ctx.Check(global_faces > 0.0,
              "IGR interior-face test found interior faces");

    if (global_faces > 0.0)
    {
      const double global_max_err = GlobalMax(local_max_err);
      ctx.AddParity(global_max_err + sigma_left + sigma_right);
      ctx.Check(global_max_err < 5.0e-11,
                "FluxWithIGR side selection matches expected sigma per side");
      ctx.Check(GlobalMax(local_base_dissipation_err) < 5.0e-11,
                "IGR sigma Rusanov toggle preserves sigma physical flux and "
                "uses base Euler dissipation");
      ctx.Check(GlobalMax(local_cfl_separation_err) < 5.0e-11,
                "IGR sigma CFL and Rusanov controls are independent");
    }
  }

  // --- Deterministic side-selection stress: explicit-side path remains stable ---
  {
    mfem::Mesh mesh_side = mfem::Mesh::MakeCartesian2D(
      /*nx=*/2,
      /*ny=*/1,
      mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    mfem::ParMesh pmesh_side(MPI_COMM_SELF, mesh_side);

    mfem::DG_FECollection fec_side(/*order=*/0, dim);
    mfem::ParFiniteElementSpace sigma_fes_side(&pmesh_side,
                                               &fec_side,
                                               /*vdim=*/1);
    mfem::ParGridFunction sigma_side(&sigma_fes_side);
    for (int e = 0; e < sigma_fes_side.GetNE(); ++e)
    {
      mfem::Array<int> vdofs;
      sigma_fes_side.GetElementVDofs(e, vdofs);
      mfem::Vector val(vdofs.Size());
      val = (e == 0) ? 0.15 : 0.45;
      sigma_side.SetSubVector(vdofs, val);
    }
    sigma_side.ExchangeFaceNbrData();

    mfem::FluxWithIGR igr_flux_side(base_flux);
    igr_flux_side.SetSigmaField(&sigma_side);

    int interior_face = -1;
    for (int f = 0; f < pmesh_side.GetNumFaces(); ++f)
    {
      mfem::FaceElementTransformations* Tr =
        pmesh_side.GetFaceElementTransformations(f);
      if (Tr && Tr->Elem1 && Tr->Elem2)
      {
        interior_face = f;
        break;
      }
    }
    MFEM_VERIFY(
      interior_face >= 0,
      "Deterministic side-selection stress test requires an interior face.");

    mfem::FaceElementTransformations* Tr =
      pmesh_side.GetFaceElementTransformations(interior_face);
    MFEM_VERIFY(Tr != nullptr,
                "Failed to obtain interior face transformations.");
    const mfem::IntegrationRule& ir_face =
      mfem::IntRules.Get(Tr->GetGeometryType(), /*order=*/1);
    const mfem::IntegrationPoint& ip_face = ir_face.IntPoint(0);
    Tr->SetAllIntPoints(&ip_face);

    mfem::Vector state = EulerConsFromPrim2D(
      /*rho=*/1.1,
      /*u=*/0.9,
      /*v=*/-0.3,
      /*p=*/1.0,
      gamma);
    mfem::Vector normal(dim);
    mfem::CalcOrtho(Tr->Jacobian(), normal);

    mfem::Vector left_ref(state.Size());
    mfem::Vector right_ref(state.Size());
    const mfem::real_t left_speed_ref =
      mfem::riemann_detail::ComputeFluxDotNForSide(igr_flux_side,
                                                   mfem::FaceSide::Left,
                                                   state,
                                                   normal,
                                                   *Tr,
                                                   left_ref);
    const mfem::real_t right_speed_ref =
      mfem::riemann_detail::ComputeFluxDotNForSide(igr_flux_side,
                                                   mfem::FaceSide::Right,
                                                   state,
                                                   normal,
                                                   *Tr,
                                                   right_ref);

    // Side regression snapshot: left/right must differ when sigma differs per element.
    const double ref_diff = std::abs(left_ref(1) - right_ref(1)) +
                            std::abs(left_ref(2) - right_ref(2)) +
                            std::abs(left_ref(3) - right_ref(3));
    ctx.AddParity(ref_diff + left_speed_ref + right_speed_ref);
    ctx.Check(ref_diff > 1.0e-12,
              "IGR side snapshots differ for left/right sigma values");

    for (int iter = 0; iter < 200; ++iter)
    {
      const mfem::FaceSide side =
        ((iter % 2) == 0) ? mfem::FaceSide::Left : mfem::FaceSide::Right;

      mfem::Vector flux_explicit(state.Size());
      const mfem::real_t speed_explicit =
        mfem::riemann_detail::ComputeFluxDotNForSide(igr_flux_side,
                                                     side,
                                                     state,
                                                     normal,
                                                     *Tr,
                                                     flux_explicit);

      const mfem::Vector& flux_ref =
        (side == mfem::FaceSide::Left) ? left_ref : right_ref;
      const mfem::real_t speed_ref =
        (side == mfem::FaceSide::Left) ? left_speed_ref : right_speed_ref;

      for (int i = 0; i < flux_ref.Size(); ++i)
      {
        ctx.CheckNear(flux_explicit(i),
                      flux_ref(i),
                      tol,
                      "Explicit-side flux remains deterministic, iter=" +
                        std::to_string(iter) +
                        ", component=" + std::to_string(i));
      }
      ctx.CheckNear(speed_explicit,
                    speed_ref,
                    tol,
                    "Explicit-side speed remains deterministic, iter=" +
                      std::to_string(iter));
    }
  }

  // --- DG backend operator path: varying sigma changes residual, sigma=0 cancels ---
  {
    constexpr int dim1 = 1;
    constexpr int num_eq1 = dim1 + 2;
    const double tol_zero = 1.0e-10;

    mfem::Mesh mesh_dg = MakePeriodicMesh1D(/*n=*/8);
    mfem::ParMesh pmesh_dg(MPI_COMM_SELF, mesh_dg);

    mfem::DG_FECollection state_fec(/*order=*/2,
                                    dim1,
                                    mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace vfes(&pmesh_dg,
                                     &state_fec,
                                     num_eq1,
                                     mfem::Ordering::byNODES);
    mfem::ParGridFunction U_const(&vfes);
    const mfem::Vector cons_const =
      EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/0.0, /*p=*/1.0, gamma);
    FillConstantState(cons_const, U_const);

    mfem::DG_FECollection sigma_fec_dg(/*order=*/1,
                                       dim1,
                                       mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace sigma_fes_dg(&pmesh_dg,
                                             &sigma_fec_dg,
                                             /*vdim=*/1,
                                             mfem::Ordering::byNODES);
    mfem::ParGridFunction sigma_dg(&sigma_fes_dg);

    class SigmaLinearCoefficient1D final : public mfem::Coefficient
    {
    public:
      double Eval(mfem::ElementTransformation& T,
                  const mfem::IntegrationPoint& ip) override
      {
        mfem::Vector x(1);
        T.Transform(ip, x);
        return 0.2 + 0.3 * x(0);
      }
    };

    SigmaLinearCoefficient1D sigma_linear;
    sigma_dg.ProjectCoefficient(sigma_linear);
    sigma_dg.ExchangeFaceNbrData();

    mfem::EulerFlux base_flux_dg(dim1, gamma);
    mfem::FluxWithIGR igr_flux_dg(base_flux_dg);
    igr_flux_dg.SetSigmaField(&sigma_dg);
    mfem::RusanovFluxWithSides num_flux_dg(igr_flux_dg);

    constexpr int int_order_offset = 2;
    auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux_dg,
                                                       int_order_offset);
    mfem::DGHyperbolicConservationLaws oper(vfes,
                                            std::move(form_integrator),
                                            /*preassembleWeakDivergence=*/true);
    oper.SetNumericalFlux(&num_flux_dg);

    mfem::Vector x_dg;
    mfem::Vector y_dg(vfes.GetTrueVSize());
    U_const.GetTrueDofs(x_dg);

    oper.Mult(x_dg, y_dg);
    const double resid_sigma = GlobalL2Norm(y_dg);

    mfem::ConstantCoefficient sigma_zero(0.0);
    sigma_dg.ProjectCoefficient(sigma_zero);
    sigma_dg.ExchangeFaceNbrData();

    oper.Mult(x_dg, y_dg);
    const double resid_zero = GlobalL2Norm(y_dg);
    ctx.AddParity(resid_sigma + resid_zero);

    ctx.Check(resid_zero < tol_zero,
              "DG (weak-div) IGR residual ~0 when sigma = 0");
    ctx.Check(resid_sigma > 1.0e-8,
              "DG (weak-div) IGR residual responds to varying sigma");

    mfem::ParGridFunction U_varying(&vfes);
    mfem::VectorFunctionCoefficient varying_state(
      num_eq1,
      [gamma](const mfem::Vector& point, mfem::Vector& value)
      {
        value = EulerConsFromPrim1D(1.0 + 0.1 * point(0),
                                    0.2 - 0.15 * point(0),
                                    0.8 + 0.2 * point(0),
                                    gamma);
      });
    U_varying.ProjectCoefficient(varying_state);
    U_varying.GetTrueDofs(x_dg);
    sigma_dg.ProjectCoefficient(sigma_linear);
    sigma_dg.ExchangeFaceNbrData();

    igr_flux_dg.SetSigmaAwareCFL(true);
    igr_flux_dg.SetSigmaAwareRusanov(true);
    mfem::Vector residual_both(vfes.GetTrueVSize());
    oper.Mult(x_dg, residual_both);
    const double speed_both = oper.GetMaxCharSpeed();

    igr_flux_dg.SetSigmaAwareRusanov(false);
    mfem::Vector residual_base_rusanov(vfes.GetTrueVSize());
    oper.Mult(x_dg, residual_base_rusanov);
    const double speed_base_rusanov = oper.GetMaxCharSpeed();

    igr_flux_dg.SetSigmaAwareRusanov(true);
    igr_flux_dg.SetSigmaAwareCFL(false);
    mfem::Vector residual_base_cfl(vfes.GetTrueVSize());
    oper.Mult(x_dg, residual_base_cfl);
    const double speed_base_cfl = oper.GetMaxCharSpeed();

    mfem::Vector rusanov_delta(residual_both);
    rusanov_delta -= residual_base_rusanov;
    mfem::Vector cfl_delta(residual_both);
    cfl_delta -= residual_base_cfl;
    ctx.Check(GlobalL2Norm(rusanov_delta) > 1.0e-10,
              "DG IGR sigma Rusanov switch changes only face dissipation");
    ctx.Check(GlobalL2Norm(cfl_delta) <=
                1.0e-11 * std::max(1.0, GlobalL2Norm(residual_both)),
              "DG IGR sigma CFL switch does not change the residual");
    ctx.CheckNear(speed_base_rusanov,
                  speed_both,
                  1.0e-12 * std::max(1.0, speed_both),
                  "DG IGR sigma Rusanov switch preserves CFL speed");
    ctx.Check(speed_base_cfl < speed_both,
              "DG IGR sigma CFL switch independently lowers CFL speed");

    igr_flux_dg.SetSigmaAwareCFL(true);
  }

  // --- IGR solver instance isolation: per-instance scratch/config must not alias ---
  {
    constexpr int igr_dim = 2;
    constexpr int igr_num_eq = igr_dim + 2;

    mfem::Mesh mesh_iso = mfem::Mesh::MakeCartesian2D(
      /*nx=*/2,
      /*ny=*/2,
      mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    mfem::ParMesh pmesh_iso(MPI_COMM_SELF, mesh_iso);

    mfem::DG_FECollection sigma_fec(/*order=*/1, igr_dim);
    mfem::DG_FECollection state_fec(/*order=*/1, igr_dim);
    mfem::ParFiniteElementSpace sigma_fes(&pmesh_iso, &sigma_fec, /*vdim=*/1);
    mfem::ParFiniteElementSpace state_fes(&pmesh_iso,
                                          &state_fec,
                                          igr_num_eq,
                                          mfem::Ordering::byNODES);

    mfem::ParGridFunction U(&state_fes);
    mfem::VectorFunctionCoefficient U_coeff(
      igr_num_eq,
      [&](const mfem::Vector& x, mfem::Vector& cons)
      {
        mfem::Vector prim(igr_num_eq);
        prim(0) = 1.0 + 0.1 * x(0);            // rho
        prim(1) = 1.5 + 0.2 * x(1);            // u
        prim(2) = -0.25 + 0.1 * x(0);          // v
        prim(1 + igr_dim) = 1.0 + 0.05 * x(1); // p
        cons.SetSize(igr_num_eq);
        mfem::EulerPrimToConservative(prim, igr_dim, gamma, cons);
      });
    U.ProjectCoefficient(U_coeff);
    U.ExchangeFaceNbrData();

    igrlab::IGRSolver solver_a(sigma_fes,
                               state_fes,
                               igr_dim,
                               igr_num_eq,
                               /*alpha=*/0.05);
    igrlab::IGRSolver solver_b(sigma_fes,
                               state_fes,
                               igr_dim,
                               igr_num_eq,
                               /*alpha=*/0.35);
    solver_a.SetRhoFloor(1.0e-12);
    solver_b.SetRhoFloor(5.0e-2);
    solver_a.SetLinearSolveParams(/*rtol=*/1.0e-10,
                                  /*atol=*/1.0e-14,
                                  /*max_it=*/200,
                                  /*print=*/0);
    solver_b.SetLinearSolveParams(/*rtol=*/1.0e-10,
                                  /*atol=*/1.0e-14,
                                  /*max_it=*/200,
                                  /*print=*/0);

    solver_a.UpdateFromState(U, /*warm_start=*/false);
    mfem::Vector sigma_a1(solver_a.GetSigma().Size());
    sigma_a1 = solver_a.GetSigma();

    solver_b.UpdateFromState(U, /*warm_start=*/false);
    mfem::Vector sigma_b(solver_b.GetSigma().Size());
    sigma_b = solver_b.GetSigma();

    solver_a.UpdateFromState(U, /*warm_start=*/false);
    mfem::Vector sigma_a2(solver_a.GetSigma().Size());
    sigma_a2 = solver_a.GetSigma();

    mfem::Vector delta_repeat(sigma_a1);
    delta_repeat -= sigma_a2;
    const double repeat_err = GlobalL2Norm(delta_repeat);

    mfem::Vector delta_cfg(sigma_a1);
    delta_cfg -= sigma_b;
    const double cfg_diff = GlobalL2Norm(delta_cfg);

    ctx.AddParity(repeat_err + cfg_diff);
    ctx.Check(
      repeat_err < 1.0e-11,
      "IGRSolver instance state is isolated across multiple solver objects");
    ctx.Check(cfg_diff > 1.0e-8,
              "Different IGRSolver configs produce distinct sigma fields");
  }

  // --- IGR linear-policy + warm-start regression ---
  {
    constexpr int dim_ws = 1;
    constexpr int num_eq_ws = dim_ws + 2;
    constexpr int nsteps = 6;

    mfem::Mesh mesh_ws = MakePeriodicMesh1D(/*n=*/12);
    mfem::ParMesh pmesh_ws(MPI_COMM_SELF, mesh_ws);

    mfem::DG_FECollection state_fec_ws(/*order=*/2,
                                       dim_ws,
                                       mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace state_fes_ws(&pmesh_ws,
                                             &state_fec_ws,
                                             num_eq_ws,
                                             mfem::Ordering::byNODES);
    mfem::ParGridFunction U_ws(&state_fes_ws);

    auto project_state = [&](double phase)
    {
      mfem::VectorFunctionCoefficient coeff(
        num_eq_ws,
        [&](const mfem::Vector& x, mfem::Vector& cons)
        {
          const double two_pi = 2.0 * std::acos(-1.0);
          const double s = std::sin(two_pi * (x(0) + phase));
          const double c = std::cos(two_pi * (x(0) + phase));
          cons = EulerConsFromPrim1D(/*rho=*/1.0 + 0.05 * s,
                                     /*u=*/0.6 + 0.05 * c,
                                     /*p=*/1.0 + 0.02 * s,
                                     gamma);
        });
      U_ws.ProjectCoefficient(coeff);
      U_ws.ExchangeFaceNbrData();
    };

    // H1 auto policy -> CG + BoomerAMG.
    mfem::H1_FECollection sigma_h1_fec(/*order=*/1, dim_ws);
    mfem::ParFiniteElementSpace sigma_h1_fes(&pmesh_ws,
                                             &sigma_h1_fec,
                                             /*vdim=*/1);
    igrlab::IGRSolver h1_solver(sigma_h1_fes,
                                state_fes_ws,
                                dim_ws,
                                num_eq_ws,
                                /*alpha=*/0.5);
    h1_solver.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
    h1_solver.SetSolveMode(igrlab::IGRSolveMode::Krylov);
    h1_solver.SetLinearSolver(igrlab::IGRLinearSolverType::Auto);
    h1_solver.SetPreconditioner(igrlab::IGRPrecondType::Auto);
    h1_solver.SetLinearSolveParams(/*rtol=*/1.0e-10,
                                   /*atol=*/0.0,
                                   /*max_it=*/250,
                                   /*print=*/0);
    project_state(/*phase=*/0.0);
    h1_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(h1_solver.GetLastLinearConverged(),
              "IGR H1 auto policy converges");
    ctx.Check(h1_solver.GetLastLinearSolverName() == "cg",
              "IGR H1 auto policy selects CG");
    ctx.Check(h1_solver.GetLastPrecondName() == "boomeramg",
              "IGR H1 auto policy selects BoomerAMG");

    // If partial assembly is forced on but a matrix-only preconditioner is
    // requested, policy should still honor the preconditioner by switching to
    // assembled-matrix path.
    {
      ScopedEnvVar force_pa("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
      igrlab::IGRSolver pa_boomer_solver(sigma_h1_fes,
                                         state_fes_ws,
                                         dim_ws,
                                         num_eq_ws,
                                         /*alpha=*/0.5);
      pa_boomer_solver.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
      pa_boomer_solver.SetSolveMode(igrlab::IGRSolveMode::Krylov);
      pa_boomer_solver.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
      pa_boomer_solver.SetPreconditioner(igrlab::IGRPrecondType::BoomerAMG);
      pa_boomer_solver.SetLinearSolveParams(/*rtol=*/1.0e-10,
                                            /*atol=*/0.0,
                                            /*max_it=*/250,
                                            /*print=*/0);
      project_state(/*phase=*/0.0);
      pa_boomer_solver.UpdateFromState(U_ws, /*warm_start=*/false);
      ctx.Check(pa_boomer_solver.GetLastLinearConverged(),
                "IGR forced-PA matrix-preconditioner policy converges");
      ctx.Check(pa_boomer_solver.GetLastPrecondName() == "boomeramg",
                "IGR forced-PA matrix-preconditioner policy keeps BoomerAMG");
      ctx.Check(pa_boomer_solver.GetAssembledOperator() != nullptr,
                "IGR forced-PA matrix-preconditioner policy keeps assembled operator");
    }

    // L2 auto policy avoids assuming production SIPG coercivity.
    mfem::DG_FECollection sigma_l2_fec(/*order=*/1,
                                       dim_ws,
                                       mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace sigma_l2_fes(&pmesh_ws,
                                             &sigma_l2_fec,
                                             /*vdim=*/1,
                                             mfem::Ordering::byNODES);
    igrlab::IGRSolver l2_solver(sigma_l2_fes,
                                state_fes_ws,
                                dim_ws,
                                num_eq_ws,
                                /*alpha=*/0.5);
    l2_solver.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
    l2_solver.SetSolveMode(igrlab::IGRSolveMode::Krylov);
    l2_solver.SetLinearSolver(igrlab::IGRLinearSolverType::Auto);
    l2_solver.SetPreconditioner(igrlab::IGRPrecondType::Auto);
    l2_solver.SetLinearSolveParams(/*rtol=*/1.0e-10,
                                   /*atol=*/0.0,
                                   /*max_it=*/250,
                                   /*print=*/0);
    project_state(/*phase=*/0.0);
    l2_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(l2_solver.GetLastLinearConverged(),
              "IGR L2 auto policy converges");
    ctx.Check(l2_solver.GetLastLinearSolverName() == "gmres",
              "IGR L2 SIPG auto policy selects GMRES");
    ctx.Check(l2_solver.GetLastPrecondName() == "boomeramg",
              "IGR L2 auto policy selects BoomerAMG");
    ctx.Check(l2_solver.GetLastAssembleSeconds() >= 0.0,
              "IGR telemetry reports non-negative assemble time");
    ctx.Check(l2_solver.GetLastFormSystemSeconds() >= 0.0,
              "IGR telemetry reports non-negative FormLinearSystem time");
    ctx.Check(l2_solver.GetLastPCSetupSeconds() >= 0.0,
              "IGR telemetry reports non-negative preconditioner setup time");
    ctx.Check(l2_solver.GetLastLinearSolveSeconds() >= 0.0,
              "IGR telemetry reports non-negative Krylov time");
    ctx.Check(l2_solver.GetTotalIGRWorkSeconds() >=
                l2_solver.GetTotalLinearSolveSeconds(),
              "IGR telemetry total includes at least Krylov time");

    // Explicit CPU QF+PA L2 reference: this exercises 2D DG face coefficient
    // projection without using GridFunction::GetVectorValue() on mesh faces.
    {
      constexpr int dim_qf = 2;
      constexpr int num_eq_qf = dim_qf + 2;
      mfem::Mesh mesh_qf =
        MakePeriodicMesh2D(/*nx=*/4, /*ny=*/3, /*sx=*/1.0, /*sy=*/0.75);
      mfem::ParMesh pmesh_qf(MPI_COMM_SELF, mesh_qf);

      mfem::DG_FECollection state_fec_qf(/*order=*/2,
                                         dim_qf,
                                         mfem::BasisType::GaussLegendre);
      mfem::ParFiniteElementSpace state_fes_qf(&pmesh_qf,
                                               &state_fec_qf,
                                               num_eq_qf,
                                               mfem::Ordering::byNODES);
      mfem::ParGridFunction U_qf(&state_fes_qf);
      mfem::VectorFunctionCoefficient U_coeff_qf(
        num_eq_qf,
        [gamma](const mfem::Vector& x, mfem::Vector& cons)
        {
          const double two_pi = 2.0 * std::acos(-1.0);
          const double sx = std::sin(two_pi * x(0));
          const double cy = std::cos(two_pi * x(1) / 0.75);
          const double rho = 1.0 + 0.06 * sx + 0.03 * cy;
          const double u = 0.7 + 0.05 * cy;
          const double v = -0.2 + 0.04 * sx;
          const double p = 1.0 + 0.02 * sx * cy;
          cons = EulerConsFromPrim2D(rho, u, v, p, gamma);
        });
      U_qf.ProjectCoefficient(U_coeff_qf);
      U_qf.ExchangeFaceNbrData();

      mfem::DG_FECollection sigma_fec_qf(/*order=*/1,
                                         dim_qf,
                                         mfem::BasisType::GaussLegendre);
      mfem::ParFiniteElementSpace sigma_fes_qf(&pmesh_qf,
                                               &sigma_fec_qf,
                                               /*vdim=*/1,
                                               mfem::Ordering::byNODES);

      {
        mfem::ParGridFunction U_side_qf(&state_fes_qf);
        U_side_qf = 0.0;
        mfem::Array<int> sdofs;
        const int state_dof = state_fes_qf.GetFE(0)->GetDof();
        for (int e = 0; e < state_fes_qf.GetNE(); ++e)
        {
          state_fes_qf.GetElementDofs(e, sdofs);
          MFEM_VERIFY(sdofs.Size() == state_dof,
                      "IGR side-density test scalar dof mismatch.");
          const double elem_rho =
            (e == 0) ? 0.05 : (0.85 + 0.09 * static_cast<double>(e));
          for (int d = 0; d < state_dof; ++d)
          {
            int rho_vdof = state_fes_qf.DofToVDof(sdofs[d], 0);
            rho_vdof = mfem::FiniteElementSpace::DecodeDof(rho_vdof);
            U_side_qf(rho_vdof) =
              elem_rho + 0.01 * static_cast<double>(d);
          }
        }
        U_side_qf.ExchangeFaceNbrData();

        const mfem::IntegrationRule& face_ir_side =
          mfem::IntRules.Get(mfem::Geometry::SEGMENT, /*order=*/5);
        mfem::FaceQuadratureSpace side_fqs(pmesh_qf,
                                           face_ir_side,
                                           mfem::FaceType::Interior);
        std::string side_reason;
        const bool side_supported =
          igrlab::dg_face::SupportsInteriorFaceSideDensityProjection2D(
            state_fes_qf,
            side_fqs,
            &side_reason);
        ctx.Check(side_supported,
                  "IGR side-density projection supports serial 2D L2 faces" +
                    (side_reason.empty() ? std::string()
                                         : (" (" + side_reason + ")")));

        if (side_supported)
        {
          igrlab::dg_face::InteriorFaceSideDensityProjectionScratch scratch;
          mfem::Vector side_q;
          const double alpha_side = 0.37;
          const double rho_floor_side = 0.2;
          const auto side_info =
            igrlab::dg_face::ProjectInteriorFaceDensityCoefficient2D(
              state_fes_qf,
              side_fqs,
              U_side_qf,
              alpha_side,
              rho_floor_side,
              side_q,
              &scratch);

          ctx.Check(side_info.qf_size == 2 * side_info.nq * side_info.nf,
                    "IGR side-density projection qdata has side/q/face layout");
          ctx.Check(side_info.nf > 0,
                    "IGR side-density projection sees interior faces");

          const auto side_vals =
            mfem::Reshape(side_q.HostRead(), 2, side_info.nq, side_info.nf);
          (void)U_side_qf.HostRead();
          mfem::Vector state_host(num_eq_qf);
          double max_side_err = 0.0;
          double max_side_jump = 0.0;

          for (int f = 0; f < side_info.nf; ++f)
          {
            const int mesh_face = side_fqs.GetMeshFaceIndex(f);
            mfem::FaceElementTransformations* Tr =
              pmesh_qf.GetFaceElementTransformations(mesh_face);
            MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                        "IGR side-density test expected interior face.");
            const mfem::IntegrationRule& ir = side_fqs.GetIntRule(f);
            for (int q = 0; q < side_info.nq; ++q)
            {
              Tr->SetAllIntPoints(&ir.IntPoint(q));
              const mfem::IntegrationPoint& ip1 =
                Tr->GetElement1IntPoint();
              const mfem::IntegrationPoint& ip2 =
                Tr->GetElement2IntPoint();

              U_side_qf.GetVectorValue(*Tr->Elem1, ip1, state_host);
              const double q1 =
                alpha_side / std::max(state_host(0), rho_floor_side);
              U_side_qf.GetVectorValue(*Tr->Elem2, ip2, state_host);
              const double q2 =
                alpha_side / std::max(state_host(0), rho_floor_side);

              max_side_err =
                std::max(max_side_err, std::abs(side_vals(0, q, f) - q1));
              max_side_err =
                std::max(max_side_err, std::abs(side_vals(1, q, f) - q2));
              max_side_jump =
                std::max(max_side_jump, std::abs(side_vals(0, q, f) -
                                                 side_vals(1, q, f)));
            }
          }

          ctx.LogMetric("igr_l2_side_density_qf max_err=" +
                        FormatSci(max_side_err) +
                        " max_side_jump=" + FormatSci(max_side_jump));
          ctx.AddParity(max_side_err + max_side_jump);
          ctx.Check(max_side_err < 5.0e-13,
                    "IGR side-density qdata matches direct left/right traces");
          ctx.Check(max_side_jump > 1.0e-3,
                    "IGR side-density qdata preserves left/right differences");
        }
      }

      // A uniform mesh cannot distinguish alpha=C_alpha*h_e^2 from a single
      // alpha=C_alpha*h_max^2.  Exercise the explicit per-element face API and
      // its constant-alpha compatibility overload on a deliberately graded
      // quad mesh.
      {
        constexpr double c_alpha = 0.41;
        constexpr double rho_value = 1.25;
        constexpr double rho_floor_local = 0.2;
        mfem::Mesh graded_mesh = MakeGradedQuadMesh2D();
        mfem::ParMesh graded_pmesh(MPI_COMM_SELF, graded_mesh);
        mfem::DG_FECollection graded_fec(/*order=*/2,
                                         dim_qf,
                                         mfem::BasisType::GaussLegendre);
        mfem::ParFiniteElementSpace graded_fes(&graded_pmesh,
                                               &graded_fec,
                                               num_eq_qf,
                                               mfem::Ordering::byNODES);
        mfem::ParGridFunction graded_state(&graded_fes);
        graded_state = 0.0;
        mfem::Array<int> graded_dofs;
        const int graded_dof = graded_fes.GetFE(0)->GetDof();
        for (int e = 0; e < graded_fes.GetNE(); ++e)
        {
          graded_fes.GetElementDofs(e, graded_dofs);
          for (int d = 0; d < graded_dof; ++d)
          {
            int vdof = graded_fes.DofToVDof(graded_dofs[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            graded_state(vdof) = rho_value;
          }
        }
        graded_state.ExchangeFaceNbrData();

        mfem::Vector element_alpha(graded_pmesh.GetNE());
        double h2_min = std::numeric_limits<double>::infinity();
        double h2_max = 0.0;
        for (int e = 0; e < graded_pmesh.GetNE(); ++e)
        {
          const double h = graded_pmesh.GetElementSize(e, 1);
          const double h2 = h * h;
          h2_min = std::min(h2_min, h2);
          h2_max = std::max(h2_max, h2);
          element_alpha(e) = c_alpha * h2;
        }
        const double max_physical_alpha = c_alpha * h2_max;

        const mfem::IntegrationRule& graded_face_ir =
          mfem::IntRules.Get(mfem::Geometry::SEGMENT, /*order=*/5);
        mfem::FaceQuadratureSpace graded_fqs(graded_pmesh,
                                             graded_face_ir,
                                             mfem::FaceType::Interior);
        igrlab::dg_face::InteriorFaceSideDensityProjectionScratch
          local_alpha_scratch;
        igrlab::dg_face::InteriorFaceSideDensityProjectionScratch
          max_alpha_scratch;
        mfem::Vector local_side_q;
        mfem::Vector max_side_q;
        const auto local_side_info =
          igrlab::dg_face::ProjectInteriorFaceDensityCoefficient2D(
            graded_fes,
            graded_fqs,
            graded_state,
            element_alpha,
            rho_floor_local,
            local_side_q,
            &local_alpha_scratch);
        const auto max_side_info =
          igrlab::dg_face::ProjectInteriorFaceDensityCoefficient2D(
            graded_fes,
            graded_fqs,
            graded_state,
            max_physical_alpha,
            rho_floor_local,
            max_side_q,
            &max_alpha_scratch);
        MFEM_VERIFY(local_side_info.nf == max_side_info.nf &&
                      local_side_info.nq == max_side_info.nq,
                    "Local/max-alpha face projections must share a layout.");

        const auto local_q = mfem::Reshape(local_side_q.HostRead(),
                                           2,
                                           local_side_info.nq,
                                           local_side_info.nf);
        const auto max_q = mfem::Reshape(max_side_q.HostRead(),
                                         2,
                                         max_side_info.nq,
                                         max_side_info.nf);
        const auto mapped_alpha =
          mfem::Reshape(local_alpha_scratch.side_alpha.HostRead(),
                        2,
                        local_side_info.nf);
        double max_local_q_error = 0.0;
        double max_constant_q_error = 0.0;
        double max_mapped_alpha_error = 0.0;
        double max_local_vs_constant = 0.0;
        double max_face_alpha_jump = 0.0;
        for (int f = 0; f < local_side_info.nf; ++f)
        {
          const int mesh_face = graded_fqs.GetMeshFaceIndex(f);
          mfem::FaceElementTransformations* Tr =
            graded_pmesh.GetFaceElementTransformations(mesh_face);
          MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                      "Graded local-alpha test expected an interior face.");
          const int elements[2] = {Tr->Elem1No, Tr->Elem2No};
          max_face_alpha_jump =
            std::max(max_face_alpha_jump,
                     std::abs(element_alpha(elements[0]) -
                              element_alpha(elements[1])));
          for (int side = 0; side < 2; ++side)
          {
            const double alpha_e = element_alpha(elements[side]);
            max_mapped_alpha_error =
              std::max(max_mapped_alpha_error,
                       std::abs(mapped_alpha(side, f) - alpha_e));
            for (int q = 0; q < local_side_info.nq; ++q)
            {
              const double expected_local = alpha_e / rho_value;
              const double expected_constant =
                max_physical_alpha / rho_value;
              max_local_q_error =
                std::max(max_local_q_error,
                         std::abs(local_q(side, q, f) - expected_local));
              max_constant_q_error =
                std::max(max_constant_q_error,
                         std::abs(max_q(side, q, f) - expected_constant));
              max_local_vs_constant =
                std::max(max_local_vs_constant,
                         std::abs(local_q(side, q, f) -
                                  max_q(side, q, f)));
            }
          }
        }

        ctx.LogMetric("igr_local_alpha_face_qf h2_ratio=" +
                      FormatSci(h2_max / h2_min) +
                      " local_err=" + FormatSci(max_local_q_error) +
                      " max_policy_err=" +
                      FormatSci(max_constant_q_error) +
                      " mapped_err=" +
                      FormatSci(max_mapped_alpha_error) +
                      " local_vs_max=" +
                      FormatSci(max_local_vs_constant));
        ctx.Check(h2_max > 2.0 * h2_min,
                  "IGR local-alpha face test uses a materially graded mesh");
        ctx.Check(max_face_alpha_jump > 1.0e-3,
                  "IGR local-alpha face test crosses unequal element sizes");
        ctx.Check(local_alpha_scratch.side_alpha.Size() ==
                    2 * local_side_info.nf &&
                    max_alpha_scratch.side_alpha.Size() == 0,
                  "IGR face projection retains explicit side alpha only for "
                  "the local policy");
        ctx.Check(max_local_q_error < 5.0e-13 &&
                    max_mapped_alpha_error < 5.0e-13,
                  "IGR local-alpha face qdata uses exact C_alpha*h_e^2 on "
                  "each side");
        ctx.Check(max_constant_q_error < 5.0e-13,
                  "IGR constant-alpha face overload remains unchanged");
        ctx.Check(max_local_vs_constant > 1.0e-3,
                  "IGR local-alpha face qdata is not alpha_max everywhere");

      }

      {
        mfem::ParGridFunction sigma_trace_qf(&sigma_fes_qf);
        sigma_trace_qf = 0.0;
        mfem::Array<int> sdofs;
        const int sigma_dof = sigma_fes_qf.GetFE(0)->GetDof();
        for (int e = 0; e < sigma_fes_qf.GetNE(); ++e)
        {
          sigma_fes_qf.GetElementDofs(e, sdofs);
          MFEM_VERIFY(sdofs.Size() == sigma_dof,
                      "IGR sigma trace test scalar dof mismatch.");
          const double elem_offset = 0.25 + 0.19 * static_cast<double>(e);
          const double elem_sign = (e % 2 == 0) ? 1.0 : -1.0;
          for (int d = 0; d < sigma_dof; ++d)
          {
            int vdof = sigma_fes_qf.DofToVDof(sdofs[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            sigma_trace_qf(vdof) =
              elem_offset + elem_sign * 0.037 * static_cast<double>(d + 1);
          }
        }
        sigma_trace_qf.ExchangeFaceNbrData();

        const mfem::IntegrationRule& face_ir_trace =
          mfem::IntRules.Get(mfem::Geometry::SEGMENT, /*order=*/5);
        mfem::FaceQuadratureSpace trace_fqs(pmesh_qf,
                                            face_ir_trace,
                                            mfem::FaceType::Interior);
        std::string trace_reason;
        const bool trace_supported =
          igrlab::dg_face::SupportsInteriorFaceScalarTraceProjection2D(
            sigma_fes_qf,
            trace_fqs,
            &trace_reason);
        ctx.Check(trace_supported,
                  "IGR scalar trace projection supports serial 2D L2 faces" +
                    (trace_reason.empty() ? std::string()
                                          : (" (" + trace_reason + ")")));

        if (trace_supported)
        {
          igrlab::dg_face::InteriorFaceScalarTraceProjectionScratch scratch;
          mfem::Vector trace_value_q;
          mfem::Vector trace_normal_q;
          const auto trace_info =
            igrlab::dg_face::ProjectInteriorFaceScalarTrace2D(
              sigma_fes_qf,
              trace_fqs,
              sigma_trace_qf,
              trace_value_q,
              trace_normal_q,
              &scratch);

          ctx.Check(trace_info.value_qf_size ==
                      2 * trace_info.nq * trace_info.nf,
                    "IGR scalar face trace qdata has side/q/face layout");
          ctx.Check(trace_info.normal_deriv_qf_size ==
                      trace_info.value_qf_size,
                    "IGR scalar face normal-derivative qdata matches trace layout");

          const auto trace_value =
            mfem::Reshape(trace_value_q.HostRead(),
                          2,
                          trace_info.nq,
                          trace_info.nf);
          const auto trace_normal =
            mfem::Reshape(trace_normal_q.HostRead(),
                          2,
                          trace_info.nq,
                          trace_info.nf);
          const double* sigma_data = sigma_trace_qf.HostRead();
          mfem::Vector shape(trace_info.dof);
          mfem::DenseMatrix dshape(trace_info.dof, dim_qf);
          mfem::Vector elem_values(trace_info.dof);
          double max_value_err = 0.0;
          double max_normal_err = 0.0;
          double max_value_jump = 0.0;
          double max_normal_mag = 0.0;

          for (int f = 0; f < trace_info.nf; ++f)
          {
            const int mesh_face = trace_fqs.GetMeshFaceIndex(f);
            const mfem::Mesh::FaceInformation face =
              pmesh_qf.GetFaceInformation(mesh_face);
            mfem::FaceElementTransformations* Tr =
              pmesh_qf.GetFaceElementTransformations(mesh_face);
            MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                        "IGR scalar trace test expected interior face.");
            const mfem::IntegrationRule& ir = trace_fqs.GetIntRule(f);
            for (int q = 0; q < trace_info.nq; ++q)
            {
              Tr->SetAllIntPoints(&ir.IntPoint(q));
              for (int side = 0; side < 2; ++side)
              {
                const int elem =
                  (side == 0) ? Tr->Elem1No : Tr->Elem2No;
                const mfem::IntegrationPoint& eip =
                  (side == 0) ? Tr->GetElement1IntPoint()
                              : Tr->GetElement2IntPoint();
                const mfem::FiniteElement& fe = *sigma_fes_qf.GetFE(elem);
                sigma_fes_qf.GetElementDofs(elem, sdofs);
                MFEM_VERIFY(sdofs.Size() == trace_info.dof,
                            "IGR scalar trace test dof mismatch.");
                for (int d = 0; d < trace_info.dof; ++d)
                {
                  int vdof = sigma_fes_qf.DofToVDof(sdofs[d], 0);
                  vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
                  elem_values(d) = sigma_data[vdof];
                }
                fe.CalcShape(eip, shape);
                fe.CalcDShape(eip, dshape);
                const int local_face_id =
                  face.element[side].local_face_id;
                const int ncomp =
                  igrlab::dg_face::ReferenceNormalComponent2D(local_face_id);
                double expected_value = 0.0;
                double expected_normal = 0.0;
                for (int d = 0; d < trace_info.dof; ++d)
                {
                  expected_value += shape(d) * elem_values(d);
                  expected_normal += dshape(d, ncomp) * elem_values(d);
                }

                max_value_err =
                  std::max(max_value_err,
                           std::abs(trace_value(side, q, f) - expected_value));
                max_normal_err =
                  std::max(max_normal_err,
                           std::abs(trace_normal(side, q, f) -
                                    expected_normal));
                max_normal_mag =
                  std::max(max_normal_mag, std::abs(trace_normal(side, q, f)));
              }
              max_value_jump =
                std::max(max_value_jump,
                         std::abs(trace_value(0, q, f) -
                                  trace_value(1, q, f)));
            }
          }

          ctx.LogMetric("igr_l2_sigma_face_trace_qf max_value_err=" +
                        FormatSci(max_value_err) +
                        " max_normal_deriv_err=" +
                        FormatSci(max_normal_err) +
                        " max_value_jump=" + FormatSci(max_value_jump) +
                        " max_normal_deriv_mag=" +
                        FormatSci(max_normal_mag));
          ctx.AddParity(max_value_err + max_normal_err + max_value_jump +
                        max_normal_mag);
          ctx.Check(max_value_err < 5.0e-13,
                    "IGR scalar face trace qdata matches direct side traces");
          ctx.Check(max_normal_err < 5.0e-13,
                    "IGR scalar face normal-derivative qdata matches CalcDShape");
          ctx.Check(max_value_jump > 1.0e-3,
                    "IGR scalar face trace qdata preserves left/right jumps");
          ctx.Check(max_normal_mag > 1.0e-3,
                    "IGR scalar face normal-derivative qdata is nontrivial");
        }
      }

      {
        mfem::ParGridFunction U_face_apply(&state_fes_qf);
        U_face_apply = 0.0;
        mfem::Array<int> state_dofs;
        const int state_dof = state_fes_qf.GetFE(0)->GetDof();
        for (int e = 0; e < state_fes_qf.GetNE(); ++e)
        {
          state_fes_qf.GetElementDofs(e, state_dofs);
          MFEM_VERIFY(state_dofs.Size() == state_dof,
                      "IGR exact face apply state dof mismatch.");
          const double elem_rho = 0.72 + 0.08 * static_cast<double>(e + 1);
          for (int d = 0; d < state_dof; ++d)
          {
            int rho_vdof = state_fes_qf.DofToVDof(state_dofs[d], 0);
            rho_vdof = mfem::FiniteElementSpace::DecodeDof(rho_vdof);
            U_face_apply(rho_vdof) =
              elem_rho + 0.017 * static_cast<double>((d + 2) % 5);
          }
        }
        U_face_apply.ExchangeFaceNbrData();

        mfem::ParGridFunction sigma_face_apply(&sigma_fes_qf);
        sigma_face_apply = 0.0;
        mfem::Array<int> sigma_dofs;
        const int sigma_dof = sigma_fes_qf.GetFE(0)->GetDof();
        for (int e = 0; e < sigma_fes_qf.GetNE(); ++e)
        {
          sigma_fes_qf.GetElementDofs(e, sigma_dofs);
          MFEM_VERIFY(sigma_dofs.Size() == sigma_dof,
                      "IGR exact face apply sigma dof mismatch.");
          const double elem_bias = -0.18 + 0.13 * static_cast<double>(e);
          for (int d = 0; d < sigma_dof; ++d)
          {
            int vdof = sigma_fes_qf.DofToVDof(sigma_dofs[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            sigma_face_apply(vdof) =
              elem_bias + 0.041 * static_cast<double>(d + 1) +
              0.009 * static_cast<double>((e + d) % 3);
          }
        }
        sigma_face_apply.ExchangeFaceNbrData();

        const mfem::IntegrationRule& face_ir_apply =
          mfem::IntRules.Get(mfem::Geometry::SEGMENT, /*order=*/5);
        mfem::FaceQuadratureSpace apply_fqs(pmesh_qf,
                                            face_ir_apply,
                                            mfem::FaceType::Interior);

        const double alpha_apply = 0.43;
        const double rho_floor_apply = 0.19;
        mfem::Vector side_q_apply;
        igrlab::dg_face::InteriorFaceSideDensityProjectionScratch coeff_scratch;
        const auto coeff_info =
          igrlab::dg_face::ProjectInteriorFaceDensityCoefficient2D(
            state_fes_qf,
            apply_fqs,
            U_face_apply,
            alpha_apply,
            rho_floor_apply,
            side_q_apply,
            &coeff_scratch);

        const double ip_sigma_apply =
          static_cast<double>(igrlab::IGRIPKind::SIPG);
        const double kappa_apply = 2.75;
        mfem::Vector exact_face_y(sigma_face_apply.Size());
        exact_face_y = 0.0;
        igrlab::dg_face::InteriorFaceDGDiffusionApplyScratch apply_scratch;
        const auto apply_info =
          igrlab::dg_face::AddInteriorFaceDGDiffusionApply2D(
            sigma_fes_qf,
            apply_fqs,
            sigma_face_apply,
            side_q_apply,
            ip_sigma_apply,
            kappa_apply,
            exact_face_y,
            &apply_scratch);

        mfem::Vector deterministic_face_y0(sigma_face_apply.Size());
        mfem::Vector deterministic_face_y1(sigma_face_apply.Size());
        deterministic_face_y0 = 0.0;
        deterministic_face_y1 = 0.0;
        igrlab::dg_face::LaunchInteriorFaceDGDiffusionApply2D(
          apply_info,
          apply_scratch,
          sigma_face_apply,
          ip_sigma_apply,
          deterministic_face_y0);
        igrlab::dg_face::LaunchInteriorFaceDGDiffusionApply2D(
          apply_info,
          apply_scratch,
          sigma_face_apply,
          ip_sigma_apply,
          deterministic_face_y1);
        const bool deterministic_face_repeat =
          SameVectorBits(deterministic_face_y0, deterministic_face_y1);

        ctx.Check(coeff_info.qf_size == 2 * apply_info.nq * apply_info.nf,
                  "IGR exact face apply receives matching side-coefficient qdata");
        ctx.Check(apply_info.nf > 0,
                  "IGR exact face apply sees interior faces");
        ctx.Check(deterministic_face_repeat,
                  "IGR exact face row gather repeats bit-for-bit");

        mfem::Vector exact_face_diag(sigma_face_apply.Size());
        exact_face_diag = 0.0;
        igrlab::dg_face::AddInteriorFaceDGDiffusionDiagonal2D(
          apply_info,
          apply_scratch,
          ip_sigma_apply,
          exact_face_diag);

        AlphaOverRhoCoefficient alpha_over_rho(U_face_apply,
                                               alpha_apply,
                                               rho_floor_apply);
        mfem::DGDiffusionIntegrator mfem_face(alpha_over_rho,
                                              ip_sigma_apply,
                                              kappa_apply);
        mfem_face.SetIntegrationRule(face_ir_apply);

        mfem::Vector assembled_face_y(sigma_face_apply.Size());
        assembled_face_y = 0.0;
        mfem::Vector assembled_face_diag(sigma_face_apply.Size());
        assembled_face_diag = 0.0;
        const double* sigma_data = sigma_face_apply.HostRead();
        double* assembled_data = assembled_face_y.HostReadWrite();
        double* assembled_diag_data = assembled_face_diag.HostReadWrite();
        mfem::DenseMatrix elmat;
        mfem::Vector xloc(2 * sigma_dof);
        mfem::Vector yloc(2 * sigma_dof);

        for (int f = 0; f < apply_fqs.GetNE(); ++f)
        {
          const int mesh_face = apply_fqs.GetMeshFaceIndex(f);
          mfem::FaceElementTransformations* Tr =
            pmesh_qf.GetFaceElementTransformations(mesh_face);
          MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                      "IGR exact face apply test expected interior face.");
          const mfem::FiniteElement& fe1 = *sigma_fes_qf.GetFE(Tr->Elem1No);
          const mfem::FiniteElement& fe2 = *sigma_fes_qf.GetFE(Tr->Elem2No);
          mfem_face.AssembleFaceMatrix(fe1, fe2, *Tr, elmat);

          sigma_fes_qf.GetElementDofs(Tr->Elem1No, sigma_dofs);
          MFEM_VERIFY(sigma_dofs.Size() == sigma_dof,
                      "IGR exact face apply side-1 dof mismatch.");
          for (int d = 0; d < sigma_dof; ++d)
          {
            int vdof = sigma_fes_qf.DofToVDof(sigma_dofs[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            xloc(d) = sigma_data[vdof];
          }
          sigma_fes_qf.GetElementDofs(Tr->Elem2No, sigma_dofs);
          MFEM_VERIFY(sigma_dofs.Size() == sigma_dof,
                      "IGR exact face apply side-2 dof mismatch.");
          for (int d = 0; d < sigma_dof; ++d)
          {
            int vdof = sigma_fes_qf.DofToVDof(sigma_dofs[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            xloc(sigma_dof + d) = sigma_data[vdof];
          }

          elmat.Mult(xloc, yloc);

          sigma_fes_qf.GetElementDofs(Tr->Elem1No, sigma_dofs);
          for (int d = 0; d < sigma_dof; ++d)
          {
            int vdof = sigma_fes_qf.DofToVDof(sigma_dofs[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            assembled_data[vdof] += yloc(d);
            assembled_diag_data[vdof] += elmat(d, d);
          }
          sigma_fes_qf.GetElementDofs(Tr->Elem2No, sigma_dofs);
          for (int d = 0; d < sigma_dof; ++d)
          {
            int vdof = sigma_fes_qf.DofToVDof(sigma_dofs[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            assembled_data[vdof] += yloc(sigma_dof + d);
            assembled_diag_data[vdof] +=
              elmat(sigma_dof + d, sigma_dof + d);
          }
        }

        mfem::Vector face_delta(exact_face_y);
        face_delta -= assembled_face_y;
        const double ref_norm =
          std::max(GlobalL2Norm(assembled_face_y), 1.0e-30);
        const double rel_err = GlobalL2Norm(face_delta) / ref_norm;
        const double* delta_data = face_delta.HostRead();
        double max_abs_err = 0.0;
        for (int i = 0; i < face_delta.Size(); ++i)
        {
          max_abs_err = std::max(max_abs_err, std::abs(delta_data[i]));
        }

        ctx.LogMetric("igr_l2_exact_face_apply_qf rel=" +
                      FormatSci(rel_err) +
                      " max_abs=" + FormatSci(max_abs_err) +
                      " ref_norm=" + FormatSci(ref_norm));
        ctx.AddParity(rel_err + max_abs_err);
        ctx.Check(rel_err < 5.0e-13,
                  "IGR exact face apply matches assembled MFEM face action");
        ctx.Check(max_abs_err < 5.0e-13,
                  "IGR exact face apply max error is roundoff");

        mfem::Vector diag_delta(exact_face_diag);
        diag_delta -= assembled_face_diag;
        const double diag_ref_norm =
          std::max(GlobalL2Norm(assembled_face_diag), 1.0e-30);
        const double diag_rel_err = GlobalL2Norm(diag_delta) / diag_ref_norm;
        const double* diag_delta_data = diag_delta.HostRead();
        double diag_max_abs = 0.0;
        for (int i = 0; i < diag_delta.Size(); ++i)
        {
          diag_max_abs =
            std::max(diag_max_abs, std::abs(diag_delta_data[i]));
        }
        ctx.LogMetric("igr_l2_exact_face_diag_qf rel=" +
                      FormatSci(diag_rel_err) +
                      " max_abs=" + FormatSci(diag_max_abs) +
                      " ref_norm=" + FormatSci(diag_ref_norm));
        ctx.Check(diag_rel_err < 5.0e-13,
                  "IGR exact face diagonal matches assembled MFEM face diagonal");
        ctx.Check(diag_max_abs < 5.0e-13,
                  "IGR exact face diagonal max error is roundoff");

        mfem::Vector refreshed_side_q(side_q_apply);
        double* refreshed_q = refreshed_side_q.HostReadWrite();
        for (int i = 0; i < refreshed_side_q.Size(); ++i)
        {
          refreshed_q[i] *= 0.71 + 0.037 * static_cast<double>(i % 7);
        }

        double refresh_apply_rel_max = 0.0;
        double refresh_diag_rel_max = 0.0;
        for (const auto scaling : {
               igrlab::dg_face::DGDiffusionFaceScaling::Arithmetic,
               igrlab::dg_face::DGDiffusionFaceScaling::Max,
               igrlab::dg_face::DGDiffusionFaceScaling::Sum,
               igrlab::dg_face::DGDiffusionFaceScaling::HarmonicWeighted })
        {
          igrlab::dg_face::InteriorFaceDGDiffusionApplyScratch cached;
          const auto cached_info =
            igrlab::dg_face::BuildInteriorFaceDGDiffusionApplyData2D(
              sigma_fes_qf,
              apply_fqs,
              side_q_apply,
              kappa_apply,
              cached.shape,
              cached.weighted_normal_dshape,
              cached.penalty,
              cached.face_elements,
              cached.element_vdof_map,
              scaling,
              &cached.normal_dshape_base,
              &cached.penalty_metric);
          igrlab::dg_face::LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
            cached_info,
            cached,
            refreshed_side_q,
            kappa_apply,
            scaling);

          mfem::Vector refreshed_y(sigma_face_apply.Size());
          refreshed_y = 0.0;
          igrlab::dg_face::LaunchInteriorFaceDGDiffusionApply2D(
            cached_info,
            cached,
            sigma_face_apply,
            ip_sigma_apply,
            refreshed_y);
          mfem::Vector refreshed_diag(sigma_face_apply.Size());
          refreshed_diag = 0.0;
          igrlab::dg_face::AddInteriorFaceDGDiffusionDiagonal2D(
            cached_info,
            cached,
            ip_sigma_apply,
            refreshed_diag);

          igrlab::dg_face::InteriorFaceDGDiffusionApplyScratch rebuilt;
          const auto rebuilt_info =
            igrlab::dg_face::BuildInteriorFaceDGDiffusionApplyData2D(
              sigma_fes_qf,
              apply_fqs,
              refreshed_side_q,
              kappa_apply,
              rebuilt.shape,
              rebuilt.weighted_normal_dshape,
              rebuilt.penalty,
              rebuilt.face_elements,
              rebuilt.element_vdof_map,
              scaling);
          mfem::Vector rebuilt_y(sigma_face_apply.Size());
          rebuilt_y = 0.0;
          igrlab::dg_face::LaunchInteriorFaceDGDiffusionApply2D(
            rebuilt_info,
            rebuilt,
            sigma_face_apply,
            ip_sigma_apply,
            rebuilt_y);
          mfem::Vector rebuilt_diag(sigma_face_apply.Size());
          rebuilt_diag = 0.0;
          igrlab::dg_face::AddInteriorFaceDGDiffusionDiagonal2D(
            rebuilt_info,
            rebuilt,
            ip_sigma_apply,
            rebuilt_diag);

          refreshed_y -= rebuilt_y;
          refreshed_diag -= rebuilt_diag;
          refresh_apply_rel_max =
            std::max(refresh_apply_rel_max,
                     GlobalL2Norm(refreshed_y) /
                       std::max(GlobalL2Norm(rebuilt_y), 1.0e-30));
          refresh_diag_rel_max =
            std::max(refresh_diag_rel_max,
                     GlobalL2Norm(refreshed_diag) /
                       std::max(GlobalL2Norm(rebuilt_diag), 1.0e-30));
        }
        ctx.LogMetric("igr_l2_exact_face_refresh apply_rel_max=" +
                      FormatSci(refresh_apply_rel_max) +
                      " diag_rel_max=" +
                      FormatSci(refresh_diag_rel_max));
        ctx.Check(refresh_apply_rel_max < 2.0e-12,
                  "IGR exact face coefficient refresh matches full rebuild");
        ctx.Check(refresh_diag_rel_max < 2.0e-12,
                  "IGR exact face diagonal refresh matches full rebuild");
      }

      // Reusing an apply scratch on an equal-sized, reordered topology must
      // invalidate its derived element-to-face incidence. Counts alone cannot
      // distinguish these two packs.
      {
        mfem::Mesh mesh_incidence_a =
          mfem::Mesh::MakeCartesian2D(/*nx=*/3,
                                      /*ny=*/2,
                                      mfem::Element::QUADRILATERAL,
                                      /*generate_edges=*/true,
                                      /*sx=*/1.0,
                                      /*sy=*/0.7);
        mfem::Mesh mesh_incidence_b =
          mfem::Mesh::MakeCartesian2D(/*nx=*/3,
                                      /*ny=*/2,
                                      mfem::Element::QUADRILATERAL,
                                      /*generate_edges=*/true,
                                      /*sx=*/1.0,
                                      /*sy=*/0.7);
        mfem::Array<int> element_reordering(6);
        const int reordered_index[6] = {2, 5, 0, 4, 1, 3};
        for (int e = 0; e < element_reordering.Size(); ++e)
        {
          element_reordering[e] = reordered_index[e];
        }
        mesh_incidence_b.ReorderElements(element_reordering,
                                         /*reorder_vertices=*/false);

        mfem::ParMesh pmesh_incidence_a(MPI_COMM_SELF, mesh_incidence_a);
        mfem::ParMesh pmesh_incidence_b(MPI_COMM_SELF, mesh_incidence_b);
        mfem::DG_FECollection incidence_fec(
          /*order=*/1, dim_qf, mfem::BasisType::GaussLegendre);
        mfem::ParFiniteElementSpace incidence_fes_a(
          &pmesh_incidence_a,
          &incidence_fec,
          /*vdim=*/1,
          mfem::Ordering::byNODES);
        mfem::ParFiniteElementSpace incidence_fes_b(
          &pmesh_incidence_b,
          &incidence_fec,
          /*vdim=*/1,
          mfem::Ordering::byNODES);
        const mfem::IntegrationRule& incidence_ir =
          mfem::IntRules.Get(mfem::Geometry::SEGMENT, /*order=*/4);
        mfem::FaceQuadratureSpace incidence_fqs_a(
          pmesh_incidence_a, incidence_ir, mfem::FaceType::Interior);
        mfem::FaceQuadratureSpace incidence_fqs_b(
          pmesh_incidence_b, incidence_ir, mfem::FaceType::Interior);

        const int incidence_nq = incidence_ir.GetNPoints();
        const int incidence_nf_a =
          igrlab::dg_face::CountLocalInteriorFaces(incidence_fes_a,
                                                   incidence_fqs_a);
        const int incidence_nf_b =
          igrlab::dg_face::CountLocalInteriorFaces(incidence_fes_b,
                                                   incidence_fqs_b);
        mfem::Vector side_q_a(2 * incidence_nq * incidence_nf_a);
        mfem::Vector side_q_b(2 * incidence_nq * incidence_nf_b);
        side_q_a = 0.37 / 1.3;
        side_q_b = 0.37 / 1.3;
        mfem::ParGridFunction sigma_a(&incidence_fes_a);
        mfem::ParGridFunction sigma_b(&incidence_fes_b);
        FillDeterministicProbeVector(sigma_a, /*seed=*/71);
        FillDeterministicProbeVector(sigma_b, /*seed=*/73);

        constexpr double incidence_ip_sigma = -1.0;
        constexpr double incidence_kappa = 2.4;
        igrlab::dg_face::InteriorFaceDGDiffusionApplyScratch reused_scratch;
        const auto incidence_info_a =
          igrlab::dg_face::BuildInteriorFaceDGDiffusionApplyData2D(
            incidence_fes_a,
            incidence_fqs_a,
            side_q_a,
            incidence_kappa,
            reused_scratch.shape,
            reused_scratch.weighted_normal_dshape,
            reused_scratch.penalty,
            reused_scratch.face_elements,
            reused_scratch.element_vdof_map);
        mfem::Vector throwaway_y(sigma_a.Size());
        throwaway_y = 0.0;
        igrlab::dg_face::LaunchInteriorFaceDGDiffusionApply2D(
          incidence_info_a,
          reused_scratch,
          sigma_a,
          incidence_ip_sigma,
          throwaway_y);
        std::vector<int> first_connectivity(
          static_cast<std::size_t>(reused_scratch.face_elements.Size()));
        const int* first_faces = reused_scratch.face_elements.HostRead();
        std::copy(first_faces,
                  first_faces + reused_scratch.face_elements.Size(),
                  first_connectivity.begin());
        const std::uint64_t first_incidence_revision =
          reused_scratch.incidence_connectivity_revision;

        const auto incidence_info_b =
          igrlab::dg_face::BuildInteriorFaceDGDiffusionApplyData2D(
            incidence_fes_b,
            incidence_fqs_b,
            side_q_b,
            incidence_kappa,
            reused_scratch.shape,
            reused_scratch.weighted_normal_dshape,
            reused_scratch.penalty,
            reused_scratch.face_elements,
            reused_scratch.element_vdof_map);
        bool packed_connectivity_changed =
          first_connectivity.size() ==
          static_cast<std::size_t>(reused_scratch.face_elements.Size());
        const int* second_faces = reused_scratch.face_elements.HostRead();
        if (packed_connectivity_changed)
        {
          packed_connectivity_changed = false;
          for (int i = 0; i < reused_scratch.face_elements.Size(); ++i)
          {
            packed_connectivity_changed =
              packed_connectivity_changed ||
              first_connectivity[static_cast<std::size_t>(i)] !=
                second_faces[i];
          }
        }
        const bool equal_sized_rebuild =
          incidence_info_a.ne == incidence_info_b.ne &&
          incidence_info_a.nf == incidence_info_b.nf &&
          incidence_info_a.dof == incidence_info_b.dof;
        const bool revision_invalidated =
          first_incidence_revision == incidence_info_a.connectivity_revision &&
          incidence_info_b.connectivity_revision !=
            incidence_info_a.connectivity_revision &&
          reused_scratch.incidence_connectivity_revision ==
            first_incidence_revision;

        mfem::Vector reused_y(sigma_b.Size());
        reused_y = 0.0;
        igrlab::dg_face::LaunchInteriorFaceDGDiffusionApply2D(
          incidence_info_b,
          reused_scratch,
          sigma_b,
          incidence_ip_sigma,
          reused_y);
        const bool incidence_revision_refreshed =
          reused_scratch.incidence_connectivity_revision ==
          incidence_info_b.connectivity_revision;

        igrlab::dg_face::InteriorFaceDGDiffusionApplyScratch fresh_scratch;
        const auto fresh_info =
          igrlab::dg_face::BuildInteriorFaceDGDiffusionApplyData2D(
            incidence_fes_b,
            incidence_fqs_b,
            side_q_b,
            incidence_kappa,
            fresh_scratch.shape,
            fresh_scratch.weighted_normal_dshape,
            fresh_scratch.penalty,
            fresh_scratch.face_elements,
            fresh_scratch.element_vdof_map);
        mfem::Vector fresh_y(sigma_b.Size());
        fresh_y = 0.0;
        igrlab::dg_face::LaunchInteriorFaceDGDiffusionApply2D(
          fresh_info,
          fresh_scratch,
          sigma_b,
          incidence_ip_sigma,
          fresh_y);
        mfem::Vector reused_repeat_y(sigma_b.Size());
        reused_repeat_y = 0.0;
        igrlab::dg_face::LaunchInteriorFaceDGDiffusionApply2D(
          incidence_info_b,
          reused_scratch,
          sigma_b,
          incidence_ip_sigma,
          reused_repeat_y);

        ctx.Check(equal_sized_rebuild && packed_connectivity_changed,
                  "IGR incidence cache regression uses equal counts and "
                  "different packed connectivity");
        ctx.Check(revision_invalidated && incidence_revision_refreshed,
                  "IGR incidence cache revision invalidates on connectivity "
                  "rebuild");
        ctx.Check(SameVectorBits(reused_y, fresh_y),
                  "IGR reordered scratch reuse matches a fresh incidence "
                  "build bit-for-bit");
        ctx.Check(SameVectorBits(reused_y, reused_repeat_y),
                  "IGR reordered incidence gather remains repeatable");
      }

      {
        mfem::Mesh mesh_orient =
          mfem::Mesh::MakeCartesian2D(/*nx=*/2,
                                      /*ny=*/1,
                                      mfem::Element::QUADRILATERAL,
                                      /*generate_edges=*/true,
                                      /*sx=*/1.0,
                                      /*sy=*/0.7);
        for (int v = 0; v < mesh_orient.GetNV(); ++v)
        {
          double* xv = mesh_orient.GetVertex(v);
          const double x0 = xv[0];
          const double y0 = xv[1];
          xv[0] = x0 + 0.11 * y0 + 0.025 * x0 * y0;
          xv[1] = y0 + 0.06 * x0 * (1.0 - y0);
        }
        {
          mfem::Element* el0 = mesh_orient.GetElement(0);
          int* v = el0->GetVertices();
          const int tmp = v[0];
          v[0] = v[1];
          v[1] = v[2];
          v[2] = v[3];
          v[3] = tmp;
        }
        {
          mfem::Element* el1 = mesh_orient.GetElement(1);
          int* v = el1->GetVertices();
          const int v0 = v[0];
          const int v1 = v[1];
          const int v2 = v[2];
          const int v3 = v[3];
          v[0] = v2;
          v[1] = v3;
          v[2] = v0;
          v[3] = v1;
        }
        mesh_orient.FinalizeTopology();
        mesh_orient.Finalize(/*refine=*/false, /*fix_orientation=*/false);

        mfem::ParMesh pmesh_orient(MPI_COMM_SELF, mesh_orient);
        mfem::DG_FECollection state_fec_orient(/*order=*/2,
                                               dim_qf,
                                               mfem::BasisType::GaussLegendre);
        mfem::ParFiniteElementSpace state_fes_orient(&pmesh_orient,
                                                     &state_fec_orient,
                                                     num_eq_qf,
                                                     mfem::Ordering::byNODES);
        mfem::DG_FECollection sigma_fec_orient(/*order=*/2,
                                               dim_qf,
                                               mfem::BasisType::GaussLegendre);
        mfem::ParFiniteElementSpace sigma_fes_orient(&pmesh_orient,
                                                     &sigma_fec_orient,
                                                     /*vdim=*/1,
                                                     mfem::Ordering::byNODES);

        mfem::ParGridFunction U_orient(&state_fes_orient);
        U_orient = 0.0;
        mfem::Array<int> state_dofs_orient;
        const int state_dof_orient = state_fes_orient.GetFE(0)->GetDof();
        for (int e = 0; e < state_fes_orient.GetNE(); ++e)
        {
          state_fes_orient.GetElementDofs(e, state_dofs_orient);
          for (int d = 0; d < state_dof_orient; ++d)
          {
            for (int c = 0; c < num_eq_qf; ++c)
            {
              int vdof = state_fes_orient.DofToVDof(state_dofs_orient[d], c);
              vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
              if (c == 0)
              {
                U_orient(vdof) =
                  0.82 + 0.06 * static_cast<double>(e) +
                  0.013 * static_cast<double>(d + 1);
              }
              else
              {
                U_orient(vdof) =
                  0.03 * static_cast<double>(c) +
                  0.007 * static_cast<double>((e + 2 * d + c) % 5);
              }
            }
          }
        }
        U_orient.ExchangeFaceNbrData();

        mfem::ParGridFunction sigma_orient(&sigma_fes_orient);
        sigma_orient = 0.0;
        mfem::Array<int> sigma_dofs_orient;
        const int sigma_dof_orient = sigma_fes_orient.GetFE(0)->GetDof();
        for (int e = 0; e < sigma_fes_orient.GetNE(); ++e)
        {
          sigma_fes_orient.GetElementDofs(e, sigma_dofs_orient);
          for (int d = 0; d < sigma_dof_orient; ++d)
          {
            int vdof = sigma_fes_orient.DofToVDof(sigma_dofs_orient[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            sigma_orient(vdof) =
              -0.09 + 0.05 * static_cast<double>(e + 1) +
              0.021 * static_cast<double>(d + 1) +
              0.004 * static_cast<double>((3 * e + d) % 4);
          }
        }
        sigma_orient.ExchangeFaceNbrData();

        const mfem::IntegrationRule& orient_face_ir =
          mfem::IntRules.Get(mfem::Geometry::SEGMENT, /*order=*/7);
        const auto orient_result =
          CompareExactFaceApplyWithAssembledMFEM(pmesh_orient,
                                                 state_fes_orient,
                                                 sigma_fes_orient,
                                                 U_orient,
                                                 sigma_orient,
                                                 orient_face_ir,
                                                 /*alpha=*/0.51,
                                                 /*rho_floor=*/0.2,
                                                 static_cast<double>(
                                                   igrlab::IGRIPKind::SIPG),
                                                 /*kappa=*/3.25);

        ctx.LogMetric("igr_l2_exact_face_apply_oriented_qf rel=" +
                      FormatSci(orient_result.rel_err) +
                      " max_abs=" + FormatSci(orient_result.max_abs_err) +
                      " ref_norm=" + FormatSci(orient_result.ref_norm) +
                      " nf=" + std::to_string(orient_result.nf) +
                      " nq=" + std::to_string(orient_result.nq));
        ctx.AddParity(orient_result.rel_err + orient_result.max_abs_err +
                      static_cast<double>(orient_result.nf));
        ctx.Check(orient_result.nf > 0,
                  "IGR exact face oriented apply sees interior faces");
        ctx.Check(orient_result.rel_err < 5.0e-13,
                  "IGR exact face oriented apply matches assembled MFEM");
        ctx.Check(orient_result.max_abs_err < 5.0e-13,
                  "IGR exact face oriented apply max error is roundoff");
      }

      auto configure_qf_solver =
        [&](igrlab::IGRSolver& solver)
      {
        solver.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
        solver.SetIPKind(igrlab::IGRIPKind::SIPG);
        solver.SetSolveMode(igrlab::IGRSolveMode::Krylov);
        solver.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
        solver.SetPreconditioner(igrlab::IGRPrecondType::None);
        solver.SetLinearSolveParams(/*rtol=*/1.0e-11,
                                    /*atol=*/0.0,
                                    /*max_it=*/1000,
                                    /*print=*/0);
      };

      mfem::H1_FECollection sigma_h1_fec_qf(/*order=*/1, dim_qf);
      mfem::ParFiniteElementSpace sigma_h1_fes_qf(&pmesh_qf,
                                                  &sigma_h1_fec_qf,
                                                  /*vdim=*/1);
      igrlab::IGRSolver assembled_h1_qf_ref(sigma_h1_fes_qf,
                                            state_fes_qf,
                                            dim_qf,
                                            num_eq_qf,
                                            /*alpha=*/0.3);
      configure_qf_solver(assembled_h1_qf_ref);
      assembled_h1_qf_ref.UpdateFromState(U_qf, /*warm_start=*/false);
	      {
	        ScopedEnvVar force_pa("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
	        ScopedEnvVar force_qf("DG_IGR_IGR_DEVICE_QF", "1");
        igrlab::IGRSolver pa_h1_qf_ref(sigma_h1_fes_qf,
                                       state_fes_qf,
                                       dim_qf,
                                       num_eq_qf,
                                       /*alpha=*/0.3);
        configure_qf_solver(pa_h1_qf_ref);
        pa_h1_qf_ref.UpdateFromState(U_qf, /*warm_start=*/false);

        mfem::Vector delta_h1_qf(assembled_h1_qf_ref.GetSigma());
        delta_h1_qf -= pa_h1_qf_ref.GetSigma();
        const double h1_ref_norm =
          std::max(GlobalL2Norm(assembled_h1_qf_ref.GetSigma()), 1.0e-30);
        const double h1_pa_qf_rel = GlobalL2Norm(delta_h1_qf) / h1_ref_norm;
        ctx.LogMetric("igr_h1_cpu_pa_qf_rel=" + FormatSci(h1_pa_qf_rel) +
                      " assembled_iters=" +
                      std::to_string(assembled_h1_qf_ref.GetLastLinearIters()) +
                      " pa_qf_iters=" +
                      std::to_string(pa_h1_qf_ref.GetLastLinearIters()));
        ctx.AddParity(h1_pa_qf_rel);
      }

      igrlab::IGRSolver assembled_qf_ref(sigma_fes_qf,
                                         state_fes_qf,
                                         dim_qf,
                                         num_eq_qf,
                                         /*alpha=*/0.3);
      configure_qf_solver(assembled_qf_ref);
      assembled_qf_ref.UpdateFromState(U_qf, /*warm_start=*/false);
      ctx.Check(assembled_qf_ref.GetLastLinearConverged(),
                "IGR assembled L2 reference solve converges");
      ctx.Check(assembled_qf_ref.GetAssembledOperator() != nullptr,
                "IGR assembled L2 reference uses a Hypre operator");

	      mfem::Vector sigma_assembled(assembled_qf_ref.GetSigma().Size());
	      sigma_assembled = assembled_qf_ref.GetSigma();
	      const double sigma_ref_norm =
	        std::max(GlobalL2Norm(sigma_assembled), 1.0e-30);

	      {
	        const mfem::FiniteElement& sigma_fe_qf = *sigma_fes_qf.GetFE(0);
	        const int requested_qorder =
	          std::max(1,
	                   2 * std::max(sigma_fe_qf.GetOrder(),
	                                state_fes_qf.GetFE(0)->GetOrder()));
	        const mfem::IntegrationRule& volume_ir =
	          (sigma_fe_qf.Space() == mfem::FunctionSpace::rQk)
	            ? mfem::RefinedIntRules.Get(sigma_fe_qf.GetGeomType(),
	                                         requested_qorder)
	            : mfem::IntRules.Get(sigma_fe_qf.GetGeomType(),
	                                  requested_qorder);
	        mfem::QuadratureSpace volume_qs(pmesh_qf, volume_ir);

	        mfem::QuadratureFunction invrho_qf(&volume_qs, 1);
	        mfem::QuadratureFunction ck_qf(&volume_qs, 1);
	        mfem::QuadratureFunction src_qf(&volume_qs, 1);
	        invrho_qf.SetVDim(1);
	        ck_qf.SetVDim(1);
	        src_qf.SetVDim(1);

	        const int ne_vol = volume_qs.GetNE();
	        const int nq_vol = volume_qs.GetIntRule(0).GetNPoints();
	        auto invrho_vals =
	          mfem::Reshape(invrho_qf.HostWrite(), nq_vol, ne_vol);
	        auto ck_vals = mfem::Reshape(ck_qf.HostWrite(), nq_vol, ne_vol);
	        auto src_vals = mfem::Reshape(src_qf.HostWrite(), nq_vol, ne_vol);
	        mfem::Vector state_at_q(num_eq_qf);
	        mfem::DenseMatrix grad_at_q(num_eq_qf, dim_qf);
	        BrokenDGIGRSourceCoefficient src_eval(U_qf,
	                                              dim_qf,
	                                              num_eq_qf,
	                                              /*alpha=*/0.3,
	                                              mfem::igrlab::IGRDefaults::
	                                                kRhoFloor);
	        for (int e = 0; e < ne_vol; ++e)
	        {
	          mfem::ElementTransformation* T = volume_qs.GetTransformation(e);
	          const mfem::IntegrationRule& ir = volume_qs.GetIntRule(e);
	          for (int q = 0; q < nq_vol; ++q)
	          {
	            const mfem::IntegrationPoint& ip = ir.IntPoint(q);
	            T->SetIntPoint(&ip);
	            U_qf.GetVectorValue(*T, ip, state_at_q);
	            U_qf.GetVectorGradient(*T, grad_at_q);
	            const double rho =
	              std::max(state_at_q(0),
	                       mfem::igrlab::IGRDefaults::kRhoFloor);
	            invrho_vals(q, e) = 1.0 / rho;
	            ck_vals(q, e) = 0.3 / rho;
	            src_eval.StateScratch() = state_at_q;
	            src_eval.GradientScratch() = grad_at_q;
	            src_vals(q, e) = src_eval.EvalFromStateAndGradient();
	          }
	        }

	        InverseRhoCoefficient invrho_cb(
	          U_qf,
	          mfem::igrlab::IGRDefaults::kRhoFloor);
	        AlphaOverRhoCoefficient ck_cb(
	          U_qf,
	          /*alpha=*/0.3,
	          mfem::igrlab::IGRDefaults::kRhoFloor);
	        BrokenDGIGRSourceCoefficient src_cb(
	          U_qf,
	          dim_qf,
	          num_eq_qf,
	          /*alpha=*/0.3,
	          mfem::igrlab::IGRDefaults::kRhoFloor);
	        mfem::QuadratureFunctionCoefficient invrho_qfc(invrho_qf);
	        mfem::QuadratureFunctionCoefficient ck_qfc(ck_qf);
	        mfem::QuadratureFunctionCoefficient src_qfc(src_qf);

	        mfem::Array<int> no_ess;
	        mfem::ParGridFunction x_default(&sigma_fes_qf);
	        mfem::ParGridFunction x_qf_assembled(&sigma_fes_qf);
	        mfem::ParGridFunction x_qf_pa(&sigma_fes_qf);
	        x_default = 0.0;
	        x_qf_assembled = 0.0;
	        x_qf_pa = 0.0;

	        mfem::ParBilinearForm volume_default(&sigma_fes_qf);
	        volume_default.AddDomainIntegrator(
	          new mfem::MassIntegrator(invrho_cb));
	        volume_default.AddDomainIntegrator(
	          new mfem::DiffusionIntegrator(ck_cb));
	        mfem::ParLinearForm rhs_default(&sigma_fes_qf);
	        rhs_default.AddDomainIntegrator(
	          new mfem::DomainLFIntegrator(src_cb));
	        volume_default.Assemble();
	        rhs_default.Assemble();
	        mfem::OperatorHandle A_default;
	        A_default.SetType(mfem::Operator::Hypre_ParCSR);
	        mfem::Vector X_default;
	        mfem::Vector B_default;
	        volume_default.FormLinearSystem(no_ess,
	                                        x_default,
	                                        rhs_default,
	                                        A_default,
	                                        X_default,
	                                        B_default);

	        mfem::ParBilinearForm volume_qf_assembled(&sigma_fes_qf);
	        auto* mass_qf_assembled =
	          new mfem::MassIntegrator(invrho_qfc);
	        mass_qf_assembled->SetIntRule(&volume_ir);
	        volume_qf_assembled.AddDomainIntegrator(mass_qf_assembled);
	        auto* diff_qf_assembled =
	          new mfem::DiffusionIntegrator(ck_qfc);
	        diff_qf_assembled->SetIntRule(&volume_ir);
	        volume_qf_assembled.AddDomainIntegrator(diff_qf_assembled);
	        mfem::ParLinearForm rhs_qf_assembled(&sigma_fes_qf);
	        auto* rhs_qf_assembled_integ =
	          new mfem::DomainLFIntegrator(src_qfc);
	        rhs_qf_assembled_integ->SetIntRule(&volume_ir);
	        rhs_qf_assembled.AddDomainIntegrator(rhs_qf_assembled_integ);
	        volume_qf_assembled.Assemble();
	        rhs_qf_assembled.Assemble();
	        mfem::OperatorHandle A_qf_assembled;
	        A_qf_assembled.SetType(mfem::Operator::Hypre_ParCSR);
	        mfem::Vector X_qf_assembled;
	        mfem::Vector B_qf_assembled;
	        volume_qf_assembled.FormLinearSystem(no_ess,
	                                             x_qf_assembled,
	                                             rhs_qf_assembled,
	                                             A_qf_assembled,
	                                             X_qf_assembled,
	                                             B_qf_assembled);

	        mfem::ParBilinearForm volume_qf_pa(&sigma_fes_qf);
	        volume_qf_pa.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
	        auto* mass_qf_pa = new mfem::MassIntegrator(invrho_qfc);
	        mass_qf_pa->SetIntRule(&volume_ir);
	        volume_qf_pa.AddDomainIntegrator(mass_qf_pa);
	        auto* diff_qf_pa = new mfem::DiffusionIntegrator(ck_qfc);
	        diff_qf_pa->SetIntRule(&volume_ir);
	        volume_qf_pa.AddDomainIntegrator(diff_qf_pa);
	        mfem::ParLinearForm rhs_qf_pa(&sigma_fes_qf);
	        auto* rhs_qf_pa_integ = new mfem::DomainLFIntegrator(src_qfc);
	        rhs_qf_pa_integ->SetIntRule(&volume_ir);
	        rhs_qf_pa.AddDomainIntegrator(rhs_qf_pa_integ);
	        rhs_qf_pa.UseFastAssembly(true);
	        volume_qf_pa.Assemble();
	        rhs_qf_pa.Assemble();
	        mfem::OperatorHandle A_qf_pa;
	        A_qf_pa.SetType(mfem::Operator::ANY_TYPE);
	        mfem::Vector X_qf_pa;
	        mfem::Vector B_qf_pa;
	        volume_qf_pa.FormLinearSystem(no_ess,
	                                      x_qf_pa,
	                                      rhs_qf_pa,
	                                      A_qf_pa,
	                                      X_qf_pa,
	                                      B_qf_pa);

	        const mfem::Operator* A_default_op = A_default.Ptr();
	        const mfem::Operator* A_qf_assembled_op = A_qf_assembled.Ptr();
	        const mfem::Operator* A_qf_pa_op = A_qf_pa.Ptr();
	        MFEM_VERIFY(A_default_op && A_qf_assembled_op && A_qf_pa_op,
	                    "IGR volume-only diagnostic requires formed operators.");

	        mfem::Vector volume_probe(A_default_op->Width());
	        for (int i = 0; i < volume_probe.Size(); ++i)
	        {
	          volume_probe(i) =
	            0.19 + std::sin(0.17 * static_cast<double>(i + 5));
	        }
	        mfem::Vector y_default(A_default_op->Height());
	        mfem::Vector y_qf_assembled(A_qf_assembled_op->Height());
	        mfem::Vector y_qf_pa(A_qf_pa_op->Height());
	        A_default_op->Mult(volume_probe, y_default);
	        A_qf_assembled_op->Mult(volume_probe, y_qf_assembled);
	        A_qf_pa_op->Mult(volume_probe, y_qf_pa);

	        auto rel_delta = [](const mfem::Vector& ref,
	                            const mfem::Vector& other)
	        {
	          mfem::Vector delta(ref);
	          delta -= other;
	          return GlobalL2Norm(delta) /
	                 std::max(GlobalL2Norm(ref), 1.0e-30);
	        };
	        const double default_vs_qf_assembled =
	          rel_delta(y_default, y_qf_assembled);
	        const double qf_assembled_vs_pa =
	          rel_delta(y_qf_assembled, y_qf_pa);
	        const double default_vs_pa = rel_delta(y_default, y_qf_pa);
	        const double rhs_default_vs_qf =
	          rel_delta(B_default, B_qf_assembled);
	        const double rhs_qf_assembled_vs_pa =
	          rel_delta(B_qf_assembled, B_qf_pa);

	        ctx.LogMetric(
	          "igr_l2_cpu_volume_qf_apply_rel default_vs_assembled_qf=" +
	          FormatSci(default_vs_qf_assembled) +
	          " assembled_qf_vs_pa_qf=" + FormatSci(qf_assembled_vs_pa) +
	          " default_vs_pa_qf=" + FormatSci(default_vs_pa) +
	          " rhs_default_vs_qf=" + FormatSci(rhs_default_vs_qf) +
	          " rhs_qf_assembled_vs_pa=" +
	          FormatSci(rhs_qf_assembled_vs_pa));
	        ctx.AddParity(default_vs_qf_assembled + qf_assembled_vs_pa +
	                      default_vs_pa + rhs_default_vs_qf +
	                      rhs_qf_assembled_vs_pa);
	      }

      {
        ScopedEnvVar force_pa("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
        ScopedEnvVar force_qf("DG_IGR_IGR_DEVICE_QF", "1");
        ScopedEnvVar cheb_order("DG_IGR_IGR_CHEB_ORDER", "3");
        ScopedEnvVar cheb_power_iters("DG_IGR_IGR_CHEB_POWER_ITERATIONS", "4");
        igrlab::IGRSolver cheb_qf_ref(sigma_fes_qf,
                                      state_fes_qf,
                                      dim_qf,
                                      num_eq_qf,
                                      /*alpha=*/0.3);
        cheb_qf_ref.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
        cheb_qf_ref.SetIPKind(igrlab::IGRIPKind::SIPG);
        cheb_qf_ref.SetSolveMode(igrlab::IGRSolveMode::Krylov);
        cheb_qf_ref.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
        cheb_qf_ref.SetPreconditioner(
          igrlab::IGRPrecondType::OperatorChebyshev);
        cheb_qf_ref.SetLinearSolveParams(/*rtol=*/1.0e-8,
                                         /*atol=*/0.0,
                                         /*max_it=*/1000,
                                         /*print=*/0);
        cheb_qf_ref.UpdateFromState(U_qf, /*warm_start=*/false);
        ctx.Check(cheb_qf_ref.GetLastLinearConverged(),
                  "IGR CPU L2 PA/QF Chebyshev preconditioner converges");
        ctx.Check(cheb_qf_ref.GetLastPrecondName() == "operatorchebyshev",
                  "IGR CPU L2 PA/QF reports OperatorChebyshev preconditioner");
        ctx.LogMetric("igr_l2_cpu_pa_qf_chebyshev iters=" +
                      std::to_string(cheb_qf_ref.GetLastLinearIters()) +
                      " order=3 power_iterations=4" +
                      " pc_setup_s=" +
                      FormatSci(cheb_qf_ref.GetLastPCSetupSeconds()) +
                      " solve_s=" +
                      FormatSci(cheb_qf_ref.GetLastLinearSolveSeconds()));
      }

      {
        ScopedEnvVar force_pa("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
        ScopedEnvVar force_qf("DG_IGR_IGR_DEVICE_QF", "1");
        ScopedEnvVar request_base_operator(
          "DG_IGR_IGR_CHEB_BASE_OPERATOR", "1");
        ScopedEnvVar cheb_order("DG_IGR_IGR_CHEB_ORDER", "3");
        ScopedEnvVar cheb_power_iters(
          "DG_IGR_IGR_CHEB_POWER_ITERATIONS", "4");
        igrlab::IGRSolver extrapolated_cheb_ref(sigma_fes_qf,
                                                state_fes_qf,
                                                dim_qf,
                                                num_eq_qf,
                                                /*alpha=*/0.3);
        configure_qf_solver(extrapolated_cheb_ref);
        extrapolated_cheb_ref.SetExactDGFaceOperator(true);
        extrapolated_cheb_ref.SetSigmaUpdatePolicy(
          igrlab::IGRSigmaUpdateCadence::ExtrapolatedEveryStep,
          /*interval=*/1);
        extrapolated_cheb_ref.SetPreconditioner(
          igrlab::IGRPrecondType::OperatorChebyshev);
        extrapolated_cheb_ref.SetLinearSolveParams(/*rtol=*/1.0e-8,
                                                   /*atol=*/0.0,
                                                   /*max_it=*/1000,
                                                   /*print=*/0);
        extrapolated_cheb_ref.UpdateFromState(U_qf, /*warm_start=*/false);
        ctx.Check(extrapolated_cheb_ref.GetLastLinearConverged(),
                  "IGR extrapolated exact-face full-system Chebyshev "
                  "preconditioner converges");
        ctx.Check(extrapolated_cheb_ref.GetLastExactFaceEffective(),
                  "IGR extrapolated Chebyshev regression exercises the "
                  "exact-face operator");
        ctx.Check(extrapolated_cheb_ref.GetLastPrecondName() ==
                    "operatorchebyshev",
                  "IGR extrapolated cadence ignores the legacy base-operator "
                  "environment request");
      }

      mfem::Vector extrapolated_arithmetic_sigma;
      {
        ScopedEnvVar force_pa("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
        ScopedEnvVar force_qf("DG_IGR_IGR_DEVICE_QF", "1");
        ScopedEnvVar request_arithmetic_scaling(
          "DG_IGR_IGR_IPDG_FACE_SCALING", "arithmetic");
        igrlab::IGRSolver arithmetic_scaling_ref(sigma_fes_qf,
                                                 state_fes_qf,
                                                 dim_qf,
                                                 num_eq_qf,
                                                 /*alpha=*/0.3);
        configure_qf_solver(arithmetic_scaling_ref);
        arithmetic_scaling_ref.SetExactDGFaceOperator(true);
        arithmetic_scaling_ref.SetSigmaUpdatePolicy(
          igrlab::IGRSigmaUpdateCadence::ExtrapolatedEveryStep,
          /*interval=*/1);
        arithmetic_scaling_ref.UpdateFromState(U_qf, /*warm_start=*/false);
        ctx.Check(arithmetic_scaling_ref.GetLastLinearConverged(),
                  "IGR extrapolated arithmetic face-scaling reference "
                  "converges");
        extrapolated_arithmetic_sigma = arithmetic_scaling_ref.GetSigma();
      }

      {
        ScopedEnvVar force_pa("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
        ScopedEnvVar force_qf("DG_IGR_IGR_DEVICE_QF", "1");
        ScopedEnvVar request_harmonic_scaling(
          "DG_IGR_IGR_IPDG_FACE_SCALING", "harmonic");
        igrlab::IGRSolver extrapolated_scaling_ref(sigma_fes_qf,
                                                   state_fes_qf,
                                                   dim_qf,
                                                   num_eq_qf,
                                                   /*alpha=*/0.3);
        configure_qf_solver(extrapolated_scaling_ref);
        extrapolated_scaling_ref.SetExactDGFaceOperator(true);
        extrapolated_scaling_ref.SetSigmaUpdatePolicy(
          igrlab::IGRSigmaUpdateCadence::ExtrapolatedEveryStep,
          /*interval=*/1);
        extrapolated_scaling_ref.UpdateFromState(U_qf,
                                                 /*warm_start=*/false);
        ctx.Check(extrapolated_scaling_ref.GetLastLinearConverged(),
                  "IGR extrapolated hostile face-scaling regression "
                  "converges");
        ctx.Check(extrapolated_scaling_ref.GetLastExactFaceEffective(),
                  "IGR extrapolated face-scaling regression exercises the "
                  "exact-face operator");
        mfem::Vector scaling_delta(extrapolated_scaling_ref.GetSigma());
        scaling_delta -= extrapolated_arithmetic_sigma;
        const double scaling_rel =
          GlobalL2Norm(scaling_delta) /
          std::max(GlobalL2Norm(extrapolated_arithmetic_sigma), 1.0e-30);
        ctx.LogMetric("igr_l2_extrapolated_face_scaling_override_rel=" +
                      FormatSci(scaling_rel));
        ctx.AddParity(scaling_rel);
        ctx.Check(scaling_rel < 5.0e-12,
                  "IGR extrapolated cadence ignores the legacy face-scaling "
                  "environment request");
      }

	      {
	        ScopedEnvVar force_pa("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
        ScopedEnvVar force_qf("DG_IGR_IGR_DEVICE_QF", "1");
        igrlab::IGRSolver pa_qf_ref(sigma_fes_qf,
                                    state_fes_qf,
                                    dim_qf,
                                    num_eq_qf,
                                    /*alpha=*/0.3);
        configure_qf_solver(pa_qf_ref);
        pa_qf_ref.UpdateFromState(U_qf, /*warm_start=*/false);
        ctx.Check(pa_qf_ref.GetLastLinearConverged(),
                  "IGR CPU L2 PA/QF reference solve converges");
        ctx.Check(pa_qf_ref.GetAssembledOperator() == nullptr,
                  "IGR CPU L2 PA/QF reference uses operator path");

        const mfem::Operator* assembled_op =
          assembled_qf_ref.GetLinearOperator();
        const mfem::Operator* pa_qf_op = pa_qf_ref.GetLinearOperator();
        MFEM_VERIFY(assembled_op != nullptr && pa_qf_op != nullptr,
                    "IGR L2 PA/QF diagnostic requires formed operators.");
        mfem::Vector probe(assembled_op->Width());
        for (int i = 0; i < probe.Size(); ++i)
        {
          probe(i) = 0.25 + std::sin(0.37 * static_cast<double>(i + 1));
        }
        mfem::Vector assembled_apply(assembled_op->Height());
        mfem::Vector pa_qf_apply(pa_qf_op->Height());
        assembled_op->Mult(probe, assembled_apply);
        pa_qf_op->Mult(probe, pa_qf_apply);
        mfem::Vector apply_delta(assembled_apply);
        apply_delta -= pa_qf_apply;
        const double apply_ref_norm =
          std::max(GlobalL2Norm(assembled_apply), 1.0e-30);
        const double apply_rel =
          GlobalL2Norm(apply_delta) / apply_ref_norm;

        mfem::Vector rhs_delta(assembled_qf_ref.GetLinearRHS());
        rhs_delta -= pa_qf_ref.GetLinearRHS();
        const double rhs_ref_norm =
          std::max(GlobalL2Norm(assembled_qf_ref.GetLinearRHS()), 1.0e-30);
        const double rhs_rel = GlobalL2Norm(rhs_delta) / rhs_ref_norm;
        ctx.LogMetric("igr_l2_cpu_pa_qf_apply_rel=" +
                      FormatSci(apply_rel) +
                      " rhs_rel=" + FormatSci(rhs_rel));

        mfem::Vector delta_qf(sigma_assembled);
        delta_qf -= pa_qf_ref.GetSigma();
        const double pa_qf_rel = GlobalL2Norm(delta_qf) / sigma_ref_norm;
        ctx.LogMetric("igr_l2_cpu_pa_qf_rel=" + FormatSci(pa_qf_rel) +
                      " assembled_iters=" +
                      std::to_string(assembled_qf_ref.GetLastLinearIters()) +
                      " pa_qf_iters=" +
                      std::to_string(pa_qf_ref.GetLastLinearIters()));
        ctx.AddParity(pa_qf_rel);
        const bool require_l2_pa_qf_equiv =
          igrlab::GetEnv("DG_IGR_REQUIRE_L2_PA_QF_EQUIV") != nullptr;
        if (require_l2_pa_qf_equiv)
        {
          ctx.Check(pa_qf_rel < 1.0e-9,
                    "IGR CPU L2 PA/QF matches assembled reference");
	        }
	        else
	        {
	          ctx.LogMetric(
	            "igr_l2_cpu_pa_qf_status=diagnostic_native_mfem_gl_face_gap");
	        }
	      }

	      {
	        ScopedEnvVar legacy_pa_opt_out("DG_IGR_IGR_PARTIAL_ASSEMBLY", "0");
	        // Exact side-aware faces own these dependencies; unrelated legacy
	        // PA/QF opt-outs must not make the typed closure fail at runtime.
	        ScopedEnvVar legacy_qf_opt_out("DG_IGR_IGR_DEVICE_QF", "0");
	        ScopedEnvVar force_exact("DG_IGR_IGR_EXACT_DG_FACE_OPERATOR", "1");
	        igrlab::IGRSolver exact_face_ref(sigma_fes_qf,
	                                         state_fes_qf,
	                                         dim_qf,
                                         num_eq_qf,
                                         /*alpha=*/0.3);
        configure_qf_solver(exact_face_ref);
        exact_face_ref.UpdateFromState(U_qf, /*warm_start=*/false);
        ctx.Check(exact_face_ref.GetLastLinearConverged(),
                  "IGR CPU L2 exact-face PA/QF reference solve converges");
        ctx.Check(exact_face_ref.GetAssembledOperator() == nullptr,
                  "IGR CPU L2 exact-face implies the PA/QF operator path");

	        const mfem::Operator* assembled_op =
	          assembled_qf_ref.GetLinearOperator();
	        const mfem::Operator* exact_face_op =
	          exact_face_ref.GetLinearOperator();
	        MFEM_VERIFY(assembled_op != nullptr && exact_face_op != nullptr,
	                    "IGR L2 exact-face diagnostic requires formed operators.");
	        mfem::Vector probe(assembled_op->Width());
        for (int i = 0; i < probe.Size(); ++i)
        {
          probe(i) = -0.11 + std::cos(0.29 * static_cast<double>(i + 3));
        }

        mfem::Vector probe_local(sigma_fes_qf.GetVSize());
        if (const mfem::Operator* prolong =
              sigma_fes_qf.GetProlongationMatrix())
        {
          prolong->Mult(probe, probe_local);
        }
        else
        {
          MFEM_VERIFY(probe.Size() == probe_local.Size(),
                      "IGR exact-face diagnostic true/local size mismatch.");
          probe_local = probe;
        }

        {
          const int sigma_order_qf = sigma_fes_qf.GetFE(0)->GetOrder();
          const int sigma_dof_qf = sigma_fes_qf.GetFE(0)->GetDof();
          const mfem::IntegrationRule& solver_face_ir =
            mfem::IntRules.Get(pmesh_qf.GetTypicalFaceGeometry(),
                               2 * sigma_order_qf);
          mfem::FaceQuadratureSpace solver_face_fqs(
            pmesh_qf,
            solver_face_ir,
            mfem::FaceType::Interior);

          mfem::Vector solver_side_q;
          igrlab::dg_face::InteriorFaceSideDensityProjectionScratch
            solver_coeff_scratch;
          igrlab::dg_face::ProjectInteriorFaceDensityCoefficient2D(
            state_fes_qf,
            solver_face_fqs,
            U_qf,
            /*alpha=*/0.3,
            mfem::igrlab::IGRDefaults::kRhoFloor,
            solver_side_q,
            &solver_coeff_scratch);

          const double ip_sigma_solver =
            static_cast<double>(igrlab::IGRIPKind::SIPG);
          const double kappa_solver =
            4.0 * (sigma_order_qf + 1) * (sigma_order_qf + 1);
          mfem::Vector exact_face_y(probe_local.Size());
          exact_face_y = 0.0;
          igrlab::dg_face::InteriorFaceDGDiffusionApplyScratch
            solver_face_scratch;
          igrlab::dg_face::AddInteriorFaceDGDiffusionApply2D(
            sigma_fes_qf,
            solver_face_fqs,
            probe_local,
            solver_side_q,
            ip_sigma_solver,
            kappa_solver,
            exact_face_y,
            &solver_face_scratch);

          AlphaOverRhoCoefficient alpha_over_rho_solver(
            U_qf,
            /*alpha=*/0.3,
            mfem::igrlab::IGRDefaults::kRhoFloor);
          mfem::DGDiffusionIntegrator mfem_solver_face(alpha_over_rho_solver,
                                                       ip_sigma_solver,
                                                       kappa_solver);
          mfem_solver_face.SetIntegrationRule(solver_face_ir);

          mfem::Vector assembled_face_y(probe_local.Size());
          assembled_face_y = 0.0;
          const double* probe_local_data = probe_local.HostRead();
          double* assembled_face_data = assembled_face_y.HostReadWrite();
          mfem::DenseMatrix elmat;
          mfem::Vector xloc(2 * sigma_dof_qf);
          mfem::Vector yloc(2 * sigma_dof_qf);
          mfem::Array<int> sigma_face_dofs;

          for (int f = 0; f < solver_face_fqs.GetNE(); ++f)
          {
            const int mesh_face = solver_face_fqs.GetMeshFaceIndex(f);
            mfem::FaceElementTransformations* Tr =
              pmesh_qf.GetFaceElementTransformations(mesh_face);
            MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                        "IGR exact-face solver diagnostic expected interior face.");
            const mfem::FiniteElement& fe1 =
              *sigma_fes_qf.GetFE(Tr->Elem1No);
            const mfem::FiniteElement& fe2 =
              *sigma_fes_qf.GetFE(Tr->Elem2No);
            mfem_solver_face.AssembleFaceMatrix(fe1, fe2, *Tr, elmat);

            sigma_fes_qf.GetElementDofs(Tr->Elem1No, sigma_face_dofs);
            MFEM_VERIFY(sigma_face_dofs.Size() == sigma_dof_qf,
                        "IGR exact-face solver side-1 dof mismatch.");
            for (int d = 0; d < sigma_dof_qf; ++d)
            {
              int vdof = sigma_fes_qf.DofToVDof(sigma_face_dofs[d], 0);
              vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
              xloc(d) = probe_local_data[vdof];
            }
            sigma_fes_qf.GetElementDofs(Tr->Elem2No, sigma_face_dofs);
            MFEM_VERIFY(sigma_face_dofs.Size() == sigma_dof_qf,
                        "IGR exact-face solver side-2 dof mismatch.");
            for (int d = 0; d < sigma_dof_qf; ++d)
            {
              int vdof = sigma_fes_qf.DofToVDof(sigma_face_dofs[d], 0);
              vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
              xloc(sigma_dof_qf + d) = probe_local_data[vdof];
            }

            elmat.Mult(xloc, yloc);

            sigma_fes_qf.GetElementDofs(Tr->Elem1No, sigma_face_dofs);
            for (int d = 0; d < sigma_dof_qf; ++d)
            {
              int vdof = sigma_fes_qf.DofToVDof(sigma_face_dofs[d], 0);
              vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
              assembled_face_data[vdof] += yloc(d);
            }
            sigma_fes_qf.GetElementDofs(Tr->Elem2No, sigma_face_dofs);
            for (int d = 0; d < sigma_dof_qf; ++d)
            {
              int vdof = sigma_fes_qf.DofToVDof(sigma_face_dofs[d], 0);
              vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
              assembled_face_data[vdof] += yloc(sigma_dof_qf + d);
            }
          }

          mfem::Vector solver_face_delta(exact_face_y);
          solver_face_delta -= assembled_face_y;
          const double solver_face_ref_norm =
            std::max(GlobalL2Norm(assembled_face_y), 1.0e-30);
          const double solver_face_rel =
            GlobalL2Norm(solver_face_delta) / solver_face_ref_norm;
          ctx.LogMetric("igr_l2_cpu_exact_face_solver_face_apply_rel=" +
                        FormatSci(solver_face_rel));
          ctx.Check(solver_face_rel < 5.0e-13,
                    "IGR exact-face solver-state face action matches MFEM");
        }

	        mfem::Vector assembled_apply(assembled_op->Height());
	        mfem::Vector exact_face_apply(exact_face_op->Height());
	        assembled_op->Mult(probe, assembled_apply);
	        exact_face_op->Mult(probe, exact_face_apply);

	        mfem::Vector apply_delta(assembled_apply);
	        apply_delta -= exact_face_apply;
        const double apply_ref_norm =
          std::max(GlobalL2Norm(assembled_apply), 1.0e-30);
        const double apply_rel =
          GlobalL2Norm(apply_delta) / apply_ref_norm;

        mfem::Vector rhs_delta(assembled_qf_ref.GetLinearRHS());
        rhs_delta -= exact_face_ref.GetLinearRHS();
        const double rhs_ref_norm =
          std::max(GlobalL2Norm(assembled_qf_ref.GetLinearRHS()), 1.0e-30);
        const double rhs_rel = GlobalL2Norm(rhs_delta) / rhs_ref_norm;

        mfem::Vector delta_exact(sigma_assembled);
        delta_exact -= exact_face_ref.GetSigma();
        const double exact_rel =
          GlobalL2Norm(delta_exact) / sigma_ref_norm;
        ctx.LogMetric("igr_l2_cpu_exact_face_pa_qf_apply_rel=" +
                      FormatSci(apply_rel) +
                      " rhs_rel=" + FormatSci(rhs_rel) +
                      " sigma_rel=" + FormatSci(exact_rel) +
                      " exact_iters=" +
                      std::to_string(exact_face_ref.GetLastLinearIters()));
        ctx.AddParity(apply_rel + rhs_rel + exact_rel);
        const bool require_l2_exact_face_equiv =
          igrlab::GetEnv("DG_IGR_REQUIRE_L2_EXACT_FACE_EQUIV") != nullptr;
        if (require_l2_exact_face_equiv)
        {
          ctx.Check(apply_rel < 1.0e-9,
                    "IGR CPU L2 exact-face PA/QF operator matches assembled reference");
        }
        else
        {
          ctx.LogMetric(
            "igr_l2_cpu_exact_face_status=diagnostic_pa_volume_or_composition_gap");
        }
      }
    }

    const mfem::HypreParMatrix* A_l2_sipg = l2_solver.GetAssembledOperator();
    MFEM_VERIFY(A_l2_sipg != nullptr,
                "IGR L2 SIPG SPD probe requires assembled Hypre operator.");
    const SPDProbeSummary sipg_probe = ProbeSPD(*A_l2_sipg, MPI_COMM_SELF);
    ctx.LogMetric(
      "igr_l2_sipg_spd_probe max_sym_rel=" + FormatSci(sipg_probe.max_sym_rel) +
      " min_rayleigh=" + FormatSci(sipg_probe.min_rayleigh));
    ctx.Check(sipg_probe.finite,
              "IGR L2 SIPG SPD probe operator-vector products are finite");
    ctx.Check(sipg_probe.max_sym_rel < 1.0e-10,
              "IGR L2 SIPG operator is symmetric to tolerance");
    ctx.Check(sipg_probe.min_rayleigh > 1.0e-12,
              "IGR L2 SIPG operator is positive definite on probe vectors");

    // Explicit CG request with SIPG should stay on CG (no fallback).
    l2_solver.SetIPKind(igrlab::IGRIPKind::SIPG);
    l2_solver.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
    l2_solver.SetPreconditioner(igrlab::IGRPrecondType::Auto);
    l2_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(l2_solver.GetLastLinearConverged(),
              "IGR L2 SIPG CG solve converges");
    ctx.Check(l2_solver.GetLastLinearSolverName() == "cg",
              "IGR L2 SIPG explicit CG request stays on CG");

    // Guardrail: non-symmetric IP kind must not use CG.
    l2_solver.SetIPKind(igrlab::IGRIPKind::NIPG);
    l2_solver.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
    l2_solver.SetPreconditioner(igrlab::IGRPrecondType::Auto);
    l2_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    const mfem::HypreParMatrix* A_l2_nipg = l2_solver.GetAssembledOperator();
    MFEM_VERIFY(A_l2_nipg != nullptr,
                "IGR L2 NIPG SPD probe requires assembled Hypre operator.");
    const SPDProbeSummary nipg_probe = ProbeSPD(*A_l2_nipg, MPI_COMM_SELF);
    ctx.LogMetric(
      "igr_l2_nipg_spd_probe max_sym_rel=" + FormatSci(nipg_probe.max_sym_rel) +
      " min_rayleigh=" + FormatSci(nipg_probe.min_rayleigh));
    ctx.Check(nipg_probe.finite,
              "IGR L2 NIPG SPD probe operator-vector products are finite");
    ctx.Check(nipg_probe.max_sym_rel >
                std::max(1.0e-10, 100.0 * sipg_probe.max_sym_rel),
              "IGR L2 NIPG operator is measurably non-symmetric");
    ctx.Check(l2_solver.GetLastLinearSolverName() == "gmres",
              "IGR guardrail falls back from CG to GMRES for NIPG");
    l2_solver.SetIPKind(igrlab::IGRIPKind::SIPG);

    // Guardrail: CG with ILU preconditioner on H1 should fallback preconditioner.
    h1_solver.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
    h1_solver.SetPreconditioner(igrlab::IGRPrecondType::HypreILU);
    h1_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(h1_solver.GetLastPrecondName() == "boomeramg",
              "IGR guardrail falls back from ILU to BoomerAMG for CG");

    // Warm-start regression: warm-start should reduce iteration counts after step 0.
    igrlab::IGRSolver cold_solver(sigma_l2_fes,
                                  state_fes_ws,
                                  dim_ws,
                                  num_eq_ws,
                                  /*alpha=*/0.5);
    igrlab::IGRSolver warm_solver(sigma_l2_fes,
                                  state_fes_ws,
                                  dim_ws,
                                  num_eq_ws,
                                  /*alpha=*/0.5);
    for (igrlab::IGRSolver* s : {&cold_solver, &warm_solver})
    {
      s->SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
      s->SetSolveMode(igrlab::IGRSolveMode::Krylov);
      s->SetLinearSolver(igrlab::IGRLinearSolverType::GMRES);
      s->SetPreconditioner(igrlab::IGRPrecondType::None);
      s->SetLinearSolveParams(/*rtol=*/1.0e-12,
                              /*atol=*/0.0,
                              /*max_it=*/500,
                              /*print=*/0);
    }

    // Explicit same-RHS replay check: warm-start should not require more iterations.
    project_state(/*phase=*/0.321);
    cold_solver.GetSigma() = 0.0;
    cold_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    warm_solver.UpdateFromState(U_ws, /*warm_start=*/false);

    cold_solver.GetSigma() = 0.0;
    cold_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    warm_solver.UpdateFromState(U_ws, /*warm_start=*/true);
    const double cold_replay_iters =
      static_cast<double>(cold_solver.GetLastLinearIters());
    const double warm_replay_iters =
      static_cast<double>(warm_solver.GetLastLinearIters());
    ctx.LogMetric(
      "igr_warm_start_same_rhs cold_iters=" + FormatSci(cold_replay_iters) +
      " warm_iters=" + FormatSci(warm_replay_iters));
    ctx.Check(
      warm_replay_iters <= cold_replay_iters,
      "IGR warm-start replay does not take more iterations than cold-start");
    ctx.Check(warm_solver.GetLastWarmStartEffective(),
              "IGR useful same-system warm start remains enabled");
    ctx.Check(!warm_solver.GetLastWarmStartGuardRejected(),
              "IGR default warm start is not rejected");
    ctx.Check(std::isnan(warm_solver.GetLastWarmStartResidualNorm()) &&
                std::isnan(warm_solver.GetLastZeroStartResidualNorm()),
              "IGR warm-start residual comparison is disabled by default");

    // Exercise the guard on the 2D L2 partial-assembly path used by CUDA.
    {
      ScopedEnvVar warm_start_guard("DG_IGR_IGR_WARM_START_GUARD", "1");
      ScopedEnvVar device_qf("DG_IGR_IGR_DEVICE_QF", "1");
      ScopedEnvVar partial_assembly("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
      mfem::Mesh guard_mesh = mfem::Mesh::MakeCartesian2D(
        2, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
      mfem::ParMesh guard_pmesh(MPI_COMM_SELF, guard_mesh);
      mfem::DG_FECollection guard_state_fec(
        1, 2, mfem::BasisType::GaussLegendre);
      mfem::DG_FECollection guard_sigma_fec(
        1, 2, mfem::BasisType::GaussLegendre);
      mfem::ParFiniteElementSpace guard_state_fes(
        &guard_pmesh, &guard_state_fec, 4, mfem::Ordering::byNODES);
      mfem::ParFiniteElementSpace guard_sigma_fes(
        &guard_pmesh, &guard_sigma_fec, 1, mfem::Ordering::byNODES);
      mfem::ParGridFunction guard_state(&guard_state_fes);
      mfem::VectorFunctionCoefficient guard_state_coeff(
        4,
        [&](const mfem::Vector& x, mfem::Vector& cons)
        {
          mfem::Vector prim(4);
          prim(0) = 1.0 + 0.05 * x(0);
          prim(1) = 0.6 + 0.04 * x(1);
          prim(2) = -0.1 + 0.03 * x(0);
          prim(3) = 1.0 + 0.02 * x(1);
          mfem::EulerPrimToConservative(prim, 2, gamma, cons);
        });
      guard_state.ProjectCoefficient(guard_state_coeff);
      guard_state.ExchangeFaceNbrData();

      igrlab::IGRSolver guard_control(
        guard_sigma_fes, guard_state_fes, 2, 4, 0.05);
      igrlab::IGRSolver guard_solver(
        guard_sigma_fes, guard_state_fes, 2, 4, 0.05);
      for (igrlab::IGRSolver* solver : {&guard_control, &guard_solver})
      {
        solver->SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
        solver->SetSolveMode(igrlab::IGRSolveMode::Krylov);
        solver->SetLinearSolver(igrlab::IGRLinearSolverType::GMRES);
        solver->SetPreconditioner(igrlab::IGRPrecondType::None);
        solver->SetExactDGFaceOperator(true);
        solver->SetLinearSolveParams(1.0e-10, 0.0, 500, 0);
      }
      guard_control.UpdateFromState(guard_state, /*warm_start=*/false);
      guard_solver.UpdateFromState(guard_state, /*warm_start=*/false);
      guard_solver.GetSigma() = 1.0e8;
      guard_solver.UpdateFromState(guard_state, /*warm_start=*/true);

      ctx.LogMetric(
        "igr_warm_start_guard stale_norm=" +
        FormatSci(guard_solver.GetLastWarmStartResidualNorm()) +
        " zero_norm=" +
        FormatSci(guard_solver.GetLastZeroStartResidualNorm()));
      ctx.Check(guard_solver.GetLastWarmStartGuardRejected(),
                "IGR stale oversized warm start is rejected");
      ctx.Check(!guard_solver.GetLastWarmStartEffective(),
                "IGR stale oversized warm start uses a cold solve");
      ctx.Check(std::isfinite(guard_solver.GetLastWarmStartResidualNorm()) &&
                  std::isfinite(guard_solver.GetLastZeroStartResidualNorm()) &&
                  guard_solver.GetLastWarmStartResidualNorm() >
                    guard_solver.GetLastZeroStartResidualNorm(),
                "IGR warm-start guard compares current-system residuals");
      mfem::Vector guarded_difference(guard_solver.GetSigma());
      guarded_difference -= guard_control.GetSigma();
      ctx.Check(GlobalL2Norm(guarded_difference) <=
                  1.0e-9 *
                    std::max(1.0, GlobalL2Norm(guard_control.GetSigma())),
                "IGR guarded stale warm start recovers the cold solution");
    }

    constexpr const char* true_residual_env =
      "DG_IGR_IGR_TRUE_RESIDUAL";
    constexpr const char* true_initial_residual_env =
      "DG_IGR_IGR_TRUE_INITIAL_RESIDUAL";
    const char* original_true_residual = igrlab::GetEnv(true_residual_env);
    const char* original_true_initial_residual =
      igrlab::GetEnv(true_initial_residual_env);
    const std::string original_true_residual_value =
      original_true_residual ? original_true_residual : "";
    const std::string original_true_initial_residual_value =
      original_true_initial_residual ? original_true_initial_residual : "";
    setenv(true_residual_env, "1", 1);
    setenv(true_initial_residual_env, "1", 1);

    cold_solver.GetSigma() = 0.0;
    cold_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(std::isfinite(cold_solver.GetLastLinearTrueResidualRel()),
              "IGR converged solve reports finite true relative residual");
    ctx.Check(cold_solver.GetLastLinearTrueResidualRel() <= 1.0e-10,
              "IGR converged solve satisfies independently recomputed residual");

    igrlab::IGRSolver forced_failure_solver(sigma_l2_fes,
                                            state_fes_ws,
                                            dim_ws,
                                            num_eq_ws,
                                            /*alpha=*/0.5);
    forced_failure_solver.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
    forced_failure_solver.SetSolveMode(igrlab::IGRSolveMode::Krylov);
    forced_failure_solver.SetLinearSolver(igrlab::IGRLinearSolverType::GMRES);
    forced_failure_solver.SetPreconditioner(igrlab::IGRPrecondType::None);
    forced_failure_solver.SetLinearSolveParams(/*rtol=*/1.0e-16,
                                               /*atol=*/0.0,
                                               /*max_it=*/1,
                                               /*print=*/0);
    forced_failure_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(!forced_failure_solver.GetLastLinearConverged(),
              "IGR one-iteration control reports nonconvergence");
    ctx.Check(forced_failure_solver.GetLastLinearIters() == 1,
              "IGR nonconvergence control reaches configured iteration cap");
    ctx.Check(
      std::isfinite(forced_failure_solver.GetLastLinearTrueResidualRel()),
      "IGR nonconverged solve reports finite true relative residual");
    ctx.Check(forced_failure_solver.GetLastLinearTrueResidualRel() > 1.0e-16,
              "IGR nonconverged solve remains above requested tolerance");
    ctx.Check(std::isfinite(GlobalL2Norm(forced_failure_solver.GetSigma())),
              "IGR default policy recovers a finite max-iteration sigma");

    if (original_true_residual)
    {
      setenv(true_residual_env, original_true_residual_value.c_str(), 1);
    }
    else
    {
      unsetenv(true_residual_env);
    }
    if (original_true_initial_residual)
    {
      setenv(true_initial_residual_env,
             original_true_initial_residual_value.c_str(),
             1);
    }
    else
    {
      unsetenv(true_initial_residual_env);
    }

    std::vector<double> cold_iters;
    std::vector<double> warm_iters;
    std::vector<double> cold_times;
    std::vector<double> warm_times;
    for (int step = 0; step < nsteps; ++step)
    {
      project_state(0.015 * static_cast<double>(step));

      cold_solver.GetSigma() = 0.0;
      cold_solver.UpdateFromState(U_ws, /*warm_start=*/false);
      warm_solver.UpdateFromState(U_ws, /*warm_start=*/step > 0);

      ctx.Check(cold_solver.GetLastLinearConverged(),
                "IGR cold-start solve converges at step " +
                  std::to_string(step));
      ctx.Check(warm_solver.GetLastLinearConverged(),
                "IGR warm-start solve converges at step " +
                  std::to_string(step));
      ctx.Check(std::isfinite(cold_solver.GetLastLinearFinalNorm()),
                "IGR cold-start final residual is finite at step " +
                  std::to_string(step));
      ctx.Check(std::isfinite(warm_solver.GetLastLinearFinalNorm()),
                "IGR warm-start final residual is finite at step " +
                  std::to_string(step));

      cold_iters.push_back(
        static_cast<double>(cold_solver.GetLastLinearIters()));
      warm_iters.push_back(
        static_cast<double>(warm_solver.GetLastLinearIters()));
      cold_times.push_back(cold_solver.GetLastLinearSolveSeconds());
      warm_times.push_back(warm_solver.GetLastLinearSolveSeconds());
    }

    const double cold_sum =
      std::accumulate(cold_iters.begin() + 1, cold_iters.end(), 0.0);
    const double warm_sum =
      std::accumulate(warm_iters.begin() + 1, warm_iters.end(), 0.0);
    const double denom = static_cast<double>(nsteps - 1);
    const double cold_avg = cold_sum / denom;
    const double warm_avg = warm_sum / denom;
    const double cold_time_avg =
      std::accumulate(cold_times.begin() + 1, cold_times.end(), 0.0) / denom;
    const double warm_time_avg =
      std::accumulate(warm_times.begin() + 1, warm_times.end(), 0.0) / denom;
    ctx.AddParity(cold_avg + warm_avg);
    ctx.LogMetric("igr_warm_start_avg_iters cold=" + FormatSci(cold_avg) +
                  " warm=" + FormatSci(warm_avg) +
                  " cold_time_s=" + FormatSci(cold_time_avg) +
                  " warm_time_s=" + FormatSci(warm_time_avg));
    ctx.Check(
      warm_avg <= cold_avg,
      "IGR warm-start average iteration count is not worse than cold-start");
  }

  // --- EulerWithIGR must update sigma from the current RK stage state ---
  {
    constexpr int dim_s = 1;
    constexpr int num_eq_s = dim_s + 2;
    constexpr double two_pi = 6.28318530717958647692;

    mfem::Mesh mesh_s = MakePeriodicMesh1D(/*n=*/8);
    mfem::ParMesh pmesh_s(MPI_COMM_SELF, mesh_s);

    mfem::DG_FECollection state_fec_s(/*order=*/2,
                                      dim_s,
                                      mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace state_fes_s(&pmesh_s,
                                            &state_fec_s,
                                            num_eq_s,
                                            mfem::Ordering::byNODES);
    mfem::ParGridFunction U_base(&state_fes_s);
    mfem::ParGridFunction U_stage(&state_fes_s);

    auto project_state = [&](mfem::ParGridFunction& U, double phase)
    {
      mfem::VectorFunctionCoefficient coeff(
        num_eq_s,
        [phase, two_pi, gamma](const mfem::Vector& x, mfem::Vector& cons)
        {
          const double rho = 1.0 + 0.15 * std::sin(two_pi * (x(0) + phase));
          const double u = 1.8 + 0.25 * std::cos(two_pi * (x(0) + phase));
          const double p = 1.0 + 0.12 * std::sin(two_pi * (x(0) + phase + 0.31));
          cons = EulerConsFromPrim1D(rho, u, p, gamma);
        });
      U.ProjectCoefficient(coeff);
      U.ExchangeFaceNbrData();
    };

    project_state(U_base, /*phase=*/0.00);
    project_state(U_stage, /*phase=*/0.11);

    mfem::DG_FECollection sigma_fec_s(/*order=*/1,
                                      dim_s,
                                      mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace sigma_fes_s(&pmesh_s,
                                            &sigma_fec_s,
                                            /*vdim=*/1,
                                            mfem::Ordering::byNODES);

    igrlab::IGRSolver igr_solver_s(sigma_fes_s,
                                   state_fes_s,
                                   dim_s,
                                   num_eq_s,
                                   /*alpha=*/0.25);
    igr_solver_s.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
    igr_solver_s.SetIPKind(igrlab::IGRIPKind::SIPG);
    igr_solver_s.SetSolveMode(igrlab::IGRSolveMode::Krylov);
    igr_solver_s.SetLinearSolver(igrlab::IGRLinearSolverType::Auto);
    igr_solver_s.SetPreconditioner(igrlab::IGRPrecondType::Auto);
    igr_solver_s.SetLinearSolveParams(/*rtol=*/1.0e-12,
                                      /*atol=*/0.0,
                                      /*max_it=*/400,
                                      /*print=*/0);
    igr_solver_s.SetSigmaUpdatePolicy(igrlab::IGRSigmaUpdateCadence::EveryStage,
                                      /*interval=*/1);
    igr_solver_s.SetStageRtolScale(1.0);

    mfem::EulerFlux base_flux_s(dim_s, gamma);
    mfem::FluxWithIGR igr_flux_s(base_flux_s);
    igr_flux_s.SetSigmaField(&igr_solver_s.GetSigma());
    mfem::RusanovFluxWithSides num_flux_s(igr_flux_s);

    auto form_integrator_s =
      std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux_s,
                                                       /*IntOrderOffset=*/2);
    mfem::DGHyperbolicConservationLaws op_s(state_fes_s,
                                            std::move(form_integrator_s),
                                            /*preassembleWeakDivergence=*/true);
    op_s.SetNumericalFlux(&num_flux_s);

    mfem::EulerWithIGR wrapped(op_s, igr_solver_s, U_base);
    ctx.Check(wrapped.SupportsPhysicalBoundaryFluxIntegral() &&
                op_s.SupportsPhysicalBoundaryFluxIntegral(),
              "EulerWithIGR forwards boundary-flux provider support");
    wrapped.BeginRKStep(/*step=*/0, /*t=*/0.0, /*dt=*/0.01);
    wrapped.SetRKStage(/*stage=*/0);

    mfem::Vector rhs(U_stage.Size());
    wrapped.ArmPhysicalBoundaryFluxIntegral();
    wrapped.Mult(U_stage, rhs);
    ctx.Check(wrapped.HasLastPhysicalBoundaryFluxIntegral() &&
                op_s.HasLastPhysicalBoundaryFluxIntegral() &&
                wrapped.GetLastPhysicalBoundaryFluxIntegral().Size() ==
                  num_eq_s &&
                wrapped.GetLastPhysicalBoundaryFluxIntegral().Normlinf() ==
                  0.0,
              "EulerWithIGR forwards one-shot boundary-flux capture through "
              "the periodic zero-boundary control");

    mfem::Vector sigma_wrapped(igr_solver_s.GetSigma().Size());
    sigma_wrapped = igr_solver_s.GetSigma();

    igr_solver_s.SetRKStageIndex(/*stage=*/0);
    igr_solver_s.UpdateFromState(U_stage, /*warm_start=*/false);
    mfem::Vector sigma_stage(igr_solver_s.GetSigma().Size());
    sigma_stage = igr_solver_s.GetSigma();

    igr_solver_s.SetRKStageIndex(/*stage=*/0);
    igr_solver_s.UpdateFromState(U_base, /*warm_start=*/false);
    mfem::Vector sigma_base(igr_solver_s.GetSigma().Size());
    sigma_base = igr_solver_s.GetSigma();

    mfem::Vector delta_stage(sigma_wrapped);
    delta_stage -= sigma_stage;
    const double stage_err = GlobalL2Norm(delta_stage);

    mfem::Vector delta_base(sigma_wrapped);
    delta_base -= sigma_base;
    const double stale_err = GlobalL2Norm(delta_base);

    mfem::Vector delta_stage_vs_base(sigma_stage);
    delta_stage_vs_base -= sigma_base;
    const double stage_vs_base = GlobalL2Norm(delta_stage_vs_base);

    ctx.LogMetric("igr_stage_sigma_consistency stage_err=" + FormatSci(stage_err) +
                  " stale_err=" + FormatSci(stale_err) +
                  " stage_vs_base=" + FormatSci(stage_vs_base));
    ctx.AddParity(stage_err + stale_err + stage_vs_base);

    ctx.Check(stage_vs_base > 1.0e-8,
              "IGR stage-consistency test setup distinguishes base and stage sigma");
    ctx.Check(stage_err <= 1.0e-5 * stage_vs_base,
              "EulerWithIGR updates sigma from stage state used in RHS");
    ctx.Check(stale_err >= 0.5 * stage_vs_base,
              "EulerWithIGR sigma is not sourced from persistent base state");

    // --- SSPRK(10,4) must drive all ten every-stage IGR closures ---
    {
      mfem::Vector state_before_ssprk104(U_base);
      const mfem::Vector uniform_before_ssprk104 = EulerConsFromPrim1D(
        /*rho=*/1.1,
        /*u=*/0.25,
        /*p=*/1.0,
        gamma);
      FillConstantState(uniform_before_ssprk104, U_base);
      U_base.ExchangeFaceNbrData();
      mfem::Vector source_components(num_eq_s);
      source_components(0) = 0.2;
      source_components(1) = -0.1;
      source_components(2) = 0.3;
      mfem::ParGridFunction projected_source(&state_fes_s);
      FillConstantState(source_components, projected_source);
      op_s.SetStaticSource(projected_source);
      ctx.Check(op_s.HasStaticSource(),
                "Euler+IGR SSPRK source control installs on the base DG "
                "operator without wrapping it");
      const int production_before_ssprk104 =
        igr_solver_s.GetProductionLinearSolveCalls();

      // Close the manual stage context above before handing lifecycle control
      // to a real stage-aware time integrator.
      wrapped.EndRKStep(U_base, /*t=*/0.0, /*state_generation=*/0);

      // Exercise the intermediate-stage tolerance path without retaining a
      // production solve-history API solely for this regression.
      igr_solver_s.SetStageRtolScale(4.0);
      mfem::RK3SSPWithLimiter ssprk104(
        state_fes_s, mfem::SSPRKMethod::SSPRK104);
      ssprk104.EnablePhysicalBoundaryFluxTracking();
      ssprk104.Init(wrapped);
      mfem::real_t ssprk104_time = 0.0;
      mfem::real_t ssprk104_dt = 1.0e-6;
      ssprk104.Step(U_base, ssprk104_time, ssprk104_dt);

      ctx.Check(ssprk104.GetStageCount() == 10,
                "SSPRK(10,4) reports ten RHS stages");
      ctx.Check(igr_solver_s.GetProductionLinearSolveCalls() ==
                  production_before_ssprk104 + 10,
                "SSPRK(10,4) performs one production IGR closure per stage");
      ctx.Check(ssprk104.GetStateGeneration() == 10u,
                "SSPRK(10,4) commits the accepted endpoint generation");
      ctx.Check(wrapped.GetSigmaStateGeneration() == 9u,
                "SSPRK(10,4) final RHS closure records stage generation nine");
      ctx.CheckNear(ssprk104_time,
                    ssprk104_dt,
                    10.0 * std::numeric_limits<mfem::real_t>::epsilon(),
                    "SSPRK(10,4) advances the physical time once");
      ctx.Check(ssprk104.SupportsPhysicalBoundaryFluxTracking() &&
                  ssprk104.HasCumulativePhysicalBoundaryFluxIntegral() &&
                  ssprk104.GetCumulativePhysicalBoundaryFluxIntegral().Size() ==
                    num_eq_s &&
                  ssprk104.GetCumulativePhysicalBoundaryFluxIntegral()
                      .Normlinf() == 0.0 &&
                  ssprk104
                      .GetCumulativePhysicalBoundaryFluxAbsoluteIntegral()
                      .Normlinf() == 0.0,
                "SSPRK(10,4) consumes forwarded Euler+IGR boundary captures");
      mfem::Vector expected_components(uniform_before_ssprk104);
      expected_components.Add(ssprk104_dt, source_components);
      mfem::ParGridFunction expected_grid_function(&state_fes_s);
      FillConstantState(expected_components, expected_grid_function);
      mfem::Vector source_increment_error(U_base);
      source_increment_error -= expected_grid_function;
      const double maximum_source_increment_error =
        source_increment_error.Normlinf();
      ctx.LogMetric("igr_ssprk104_static_source_max_error=" +
                    FormatSci(maximum_source_increment_error));
      ctx.Check(maximum_source_increment_error <= 2.0e-10,
                "EulerWithIGR and SSPRK(10,4) preserve the exact uniform "
                "static-source increment");

      igr_solver_s.SetStageRtolScale(1.0);
      mfem::Vector zero_source(U_base.Size());
      zero_source = 0.0;
      op_s.SetStaticSource(zero_source);
      ctx.Check(!op_s.HasStaticSource(),
                "clearing the IGR base source restores the source-free path");
      U_base = state_before_ssprk104;
      U_base.ExchangeFaceNbrData();

      // An actual Euler+IGR stage that is rejected by the callback pipeline
      // must not publish its already-captured RHS flux.
      mfem::RK3SSPWithLimiter rejected_ssprk(
        state_fes_s, mfem::SSPRKMethod::SSPRK33);
      rejected_ssprk.EnablePhysicalBoundaryFluxTracking();
      rejected_ssprk.AddAlwaysCallback(
        [](mfem::ParFiniteElementSpace&,
           mfem::ParGridFunction&,
           double)
        { throw mfem::RecoverableLimiterFailure(1); });
      rejected_ssprk.Init(wrapped);
      const mfem::Vector state_before_rejection(U_base);
      mfem::real_t rejected_time = 0.0;
      mfem::real_t rejected_dt = 1.0e-6;
      bool caught_rejection = false;
      try
      {
        rejected_ssprk.Step(U_base, rejected_time, rejected_dt);
      }
      catch (const mfem::RecoverableLimiterFailure&)
      {
        caught_rejection = true;
        U_base = state_before_rejection;
        U_base.ExchangeFaceNbrData();
        rejected_ssprk.RestoreStateGeneration(0);
        wrapped.AbortRKStep(U_base,
                           /*restored_time=*/0.0,
                           /*restored_state_generation=*/0,
                           /*retry_dt=*/0.0);
      }
      ctx.Check(caught_rejection && rejected_time == 0.0 &&
                  !rejected_ssprk
                     .HasCumulativePhysicalBoundaryFluxIntegral(),
                "rejected Euler+IGR SSPRK attempt rolls back before boundary "
                "flux commit");
    }

    // --- Sigma output lifecycle: freshness, direct reuse, and invalidation ---
    const int lifecycle_calls_start = igr_solver_s.GetLinearSolveCalls();
    const int lifecycle_production_start =
      igr_solver_s.GetProductionLinearSolveCalls();
    const int lifecycle_output_start =
      igr_solver_s.GetOutputOnlyLinearSolveCalls();
    const long long lifecycle_reuses_start =
      wrapped.GetSigmaStage0ReuseCount();
    const long long lifecycle_invalidations_start =
      wrapped.GetSigmaPrefetchInvalidationCount();
    const long long lifecycle_provider_output_start =
      wrapped.GetSigmaOutputOnlySolveCount();

    mfem::Vector state_before_prepare(U_base);
    wrapped.EndRKStep(U_base, /*t=*/0.1, /*state_generation=*/100);
    const mfem::IGRSigmaPreparationResult prepared =
      wrapped.PrepareSigmaForOutput(U_base,
                                    /*state_generation=*/100,
                                    /*next_rk_step=*/1,
                                    /*time=*/0.1,
                                    /*reusable_stage0=*/true);
    ctx.Check(prepared.solved && prepared.reusable_stage0 &&
                !prepared.output_only,
              "Sigma lifecycle: stale stage sigma is refreshed for final state");
    ctx.Check(igr_solver_s.GetLinearSolveCalls() == lifecycle_calls_start + 1,
              "Sigma lifecycle: final-state preparation performs one solve");
    mfem::Vector state_after_prepare(U_base);
    state_after_prepare -= state_before_prepare;
    ctx.CheckNear(state_after_prepare.Normlinf(),
                  0.0,
                  0.0,
                  "Sigma lifecycle: preparation leaves conservative state unchanged");

    mfem::Vector sigma_prepared(igr_solver_s.GetSigma());
    mfem::Vector prepared_error(sigma_prepared);
    prepared_error -= sigma_base;
    ctx.Check(prepared_error.Norml2() <= 1.0e-5 * stage_vs_base,
              "Sigma lifecycle: prepared sigma matches direct final-state solve");
    ctx.Check(wrapped.GetSigmaStateGeneration() == 100,
              "Sigma lifecycle: prepared sigma records state generation");

    const mfem::IGRSigmaPreparationResult prepared_again =
      wrapped.PrepareSigmaForOutput(U_base,
                                    /*state_generation=*/100,
                                    /*next_rk_step=*/1,
                                    /*time=*/0.1,
                                    /*reusable_stage0=*/true);
    ctx.Check(!prepared_again.solved,
              "Sigma lifecycle: matching live sigma is reused directly");
    ctx.Check(igr_solver_s.GetLinearSolveCalls() == lifecycle_calls_start + 1,
              "Sigma lifecycle: repeated output preparation adds no solve");

    dg_sim::EulerFieldList output_fields;
    output_fields.igr = &igr_solver_s;
    output_fields.sigma = &igr_solver_s.GetSigma();
    const int calls_before_io = igr_solver_s.GetLinearSolveCalls();
    mfem::Vector sigma_before_io(igr_solver_s.GetSigma());
    dg_sim::EulerOutputFieldTiming io_timing;
    dg_sim::UpdateEulerOutputFields(output_fields, &io_timing);
    mfem::Vector sigma_after_io(igr_solver_s.GetSigma());
    sigma_after_io -= sigma_before_io;
    ctx.Check(igr_solver_s.GetLinearSolveCalls() == calls_before_io,
              "Sigma lifecycle: output field updater invokes zero IGR solves");
    ctx.CheckNear(sigma_after_io.Normlinf(),
                  0.0,
                  0.0,
                  "Sigma lifecycle: output field updater is read-only for sigma");

    wrapped.BeginRKStep(/*step=*/1, /*t=*/0.1, /*dt=*/0.01);
    wrapped.SetRKStageState(/*stage=*/0, /*state_generation=*/100);
    wrapped.Mult(U_base, rhs);
    ctx.Check(igr_solver_s.GetLinearSolveCalls() == lifecycle_calls_start + 1,
              "Sigma lifecycle: prepared final state replaces next stage-0 solve");
    ctx.Check(wrapped.GetSigmaStage0ReuseCount() == lifecycle_reuses_start + 1,
              "Sigma lifecycle: stage-0 reuse is accounted once");
    mfem::Vector sigma_after_reuse(igr_solver_s.GetSigma());
    sigma_after_reuse -= sigma_prepared;
    ctx.CheckNear(sigma_after_reuse.Normlinf(),
                  0.0,
                  0.0,
                  "Sigma lifecycle: stage-0 reuse preserves prepared sigma exactly");

    wrapped.SetRKStageState(/*stage=*/1, /*state_generation=*/101);
    wrapped.Mult(U_stage, rhs);
    ctx.Check(igr_solver_s.GetLinearSolveCalls() == lifecycle_calls_start + 2,
              "Sigma lifecycle: distinct stage generation recomputes sigma");

    wrapped.EndRKStep(U_base, /*t=*/0.2, /*state_generation=*/102);
    wrapped.PrepareSigmaForOutput(U_base,
                                  /*state_generation=*/102,
                                  /*next_rk_step=*/2,
                                  /*time=*/0.2,
                                  /*reusable_stage0=*/true);
    const int calls_before_invalidated_stage =
      igr_solver_s.GetLinearSolveCalls();
    const int production_before_invalidated_stage =
      igr_solver_s.GetProductionLinearSolveCalls();
    mfem::Vector U_changed(U_base);
    U_changed(0) += 1.0e-6;
    wrapped.BeginRKStep(/*step=*/2, /*t=*/0.2, /*dt=*/0.01);
    wrapped.SetRKStageState(/*stage=*/0, /*state_generation=*/103);
    wrapped.Mult(U_changed, rhs);
    ctx.Check(igr_solver_s.GetLinearSolveCalls() ==
                calls_before_invalidated_stage + 1,
              "Sigma lifecycle: intervening state generation invalidates reuse");
    ctx.Check(wrapped.GetSigmaStage0ReuseCount() == lifecycle_reuses_start + 1,
              "Sigma lifecycle: invalidated stage does not count as reuse");
    ctx.Check(wrapped.GetSigmaPrefetchInvalidationCount() ==
                lifecycle_invalidations_start + 1,
              "Sigma lifecycle: changed state records one prefetch invalidation");
    ctx.Check(igr_solver_s.GetOutputOnlyLinearSolveCalls() ==
                lifecycle_output_start + 1,
              "Sigma lifecycle: invalidated prefetch becomes output-only work");
    ctx.Check(igr_solver_s.GetProductionLinearSolveCalls() ==
                production_before_invalidated_stage,
              "Sigma lifecycle: invalidated prefetch and replacement do not "
              "double count production work");

    wrapped.EndRKStep(U_stage, /*t=*/0.3, /*state_generation=*/104);
    const int production_before_output_only =
      igr_solver_s.GetProductionLinearSolveCalls();
    const mfem::IGRSigmaPreparationResult output_only =
      wrapped.PrepareSigmaForOutput(U_stage,
                                    /*state_generation=*/104,
                                    /*next_rk_step=*/3,
                                    /*time=*/0.3,
                                    /*reusable_stage0=*/false);
    ctx.Check(output_only.solved && output_only.output_only,
              "Sigma lifecycle: final output solve is explicitly classified");
    ctx.Check(igr_solver_s.GetOutputOnlyLinearSolveCalls() ==
                lifecycle_output_start + 2,
              "Sigma lifecycle: output-only solve has separate accounting");
    ctx.Check(igr_solver_s.GetProductionLinearSolveCalls() ==
                production_before_output_only,
              "Sigma lifecycle: output-only solve does not alter production count");
    ctx.Check(wrapped.GetSigmaOutputOnlySolveCount() ==
                lifecycle_provider_output_start + 2,
              "Sigma lifecycle: provider records output-only solve");
    ctx.Check(igr_solver_s.GetProductionLinearSolveCalls() ==
                lifecycle_production_start + 3,
              "Sigma lifecycle: production preparations remain production work");

    const int calls_before_cached_certificate =
      igr_solver_s.GetLinearSolveCalls();
    const int output_before_cached_certificate =
      igr_solver_s.GetOutputOnlyLinearSolveCalls();
    const mfem::IGRSigmaPreparationResult cached_certificate =
      wrapped.PrepareSigmaForOutput(U_stage,
                                    /*state_generation=*/104,
                                    /*next_rk_step=*/3,
                                    /*time=*/0.3,
                                    /*reusable_stage0=*/false,
                                    /*cold_start=*/false,
                                    /*certify_residual=*/true);
    ctx.Check(cached_certificate.solved &&
                cached_certificate.independent_residual_certified &&
                std::isfinite(
                  cached_certificate.independent_residual_rel_rhs),
              "Sigma lifecycle: certification bypasses generation-cache reuse");
    ctx.Check(igr_solver_s.GetLinearSolveCalls() ==
                  calls_before_cached_certificate + 1 &&
                igr_solver_s.GetOutputOnlyLinearSolveCalls() ==
                  output_before_cached_certificate + 1,
              "Sigma lifecycle: cached accepted state receives one independent "
              "output-only certificate solve");

    mfem::ParGridFunction U_checkpoint_next(&state_fes_s);
    project_state(U_checkpoint_next, /*phase=*/0.05);
    auto configure_checkpoint_solver = [](igrlab::IGRSolver& solver)
    {
      solver.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
      solver.SetIPKind(igrlab::IGRIPKind::SIPG);
      solver.SetSolveMode(igrlab::IGRSolveMode::Krylov);
      solver.SetLinearSolver(igrlab::IGRLinearSolverType::Auto);
      solver.SetPreconditioner(igrlab::IGRPrecondType::Auto);
      solver.SetLinearSolveParams(/*rtol=*/1.0e-12,
                                  /*atol=*/0.0,
                                  /*max_it=*/400,
                                  /*print=*/0);
    };
    igrlab::IGRSolver checkpoint_solver(sigma_fes_s,
                                        state_fes_s,
                                        dim_s,
                                        num_eq_s,
                                        /*alpha=*/0.25);
    igrlab::IGRSolver checkpoint_control(sigma_fes_s,
                                         state_fes_s,
                                         dim_s,
                                         num_eq_s,
                                         /*alpha=*/0.25);
    configure_checkpoint_solver(checkpoint_solver);
    configure_checkpoint_solver(checkpoint_control);
    checkpoint_solver.UpdateFromState(U_base, /*warm_start=*/false);
    checkpoint_control.UpdateFromState(U_base, /*warm_start=*/false);
    mfem::Vector sigma_before_abandoned_prefetch(checkpoint_solver.GetSigma());
    checkpoint_solver.UpdateFromState(
      U_stage,
      /*warm_start=*/true,
      igrlab::IGRSolvePurpose::ProductionStage0Prefetch);
    ctx.Check(checkpoint_solver.ReclassifyPendingStage0PrefetchAsOutputOnly(),
              "Sigma lifecycle: pending prefetch can be reclassified");
    mfem::Vector restored_sigma(checkpoint_solver.GetSigma());
    restored_sigma -= sigma_before_abandoned_prefetch;
    ctx.CheckNear(restored_sigma.Normlinf(),
                  0.0,
                  0.0,
                  "Sigma lifecycle: abandoned prefetch restores warm-start sigma");
    checkpoint_solver.UpdateFromState(U_checkpoint_next, /*warm_start=*/true);
    const int checkpoint_iters = checkpoint_solver.GetLastLinearIters();
    checkpoint_control.UpdateFromState(U_checkpoint_next, /*warm_start=*/true);
    mfem::Vector checkpoint_delta(checkpoint_solver.GetSigma());
    checkpoint_delta -= checkpoint_control.GetSigma();
    ctx.CheckNear(checkpoint_delta.Normlinf(),
                  0.0,
                  0.0,
                  "Sigma lifecycle: abandoned prefetch leaves production sigma exact");
    ctx.Check(checkpoint_iters == checkpoint_control.GetLastLinearIters(),
              "Sigma lifecycle: abandoned prefetch leaves production iterations exact");
    ctx.Check(checkpoint_solver.GetProductionLinearSolveCalls() == 2 &&
                checkpoint_solver.GetOutputOnlyLinearSolveCalls() == 1,
              "Sigma lifecycle: abandoned prefetch accounting is separated");

    ctx.AddParity(static_cast<double>(wrapped.GetSigmaPrefetchSolveCount() +
                                      wrapped.GetSigmaStage0ReuseCount() +
                                      wrapped.GetSigmaOutputOnlySolveCount()));

    auto check_non_stage_fallback =
      [&](igrlab::IGRSigmaUpdateCadence cadence,
          int interval,
          const std::string& label)
    {
      igrlab::IGRSolver igr_solver_fb(sigma_fes_s,
                                      state_fes_s,
                                      dim_s,
                                      num_eq_s,
                                      /*alpha=*/0.25);
      igr_solver_fb.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
      igr_solver_fb.SetIPKind(igrlab::IGRIPKind::SIPG);
      igr_solver_fb.SetSolveMode(igrlab::IGRSolveMode::Krylov);
      igr_solver_fb.SetLinearSolver(igrlab::IGRLinearSolverType::Auto);
      igr_solver_fb.SetPreconditioner(igrlab::IGRPrecondType::Auto);
      igr_solver_fb.SetLinearSolveParams(/*rtol=*/1.0e-12,
                                         /*atol=*/0.0,
                                         /*max_it=*/400,
                                         /*print=*/0);
      igr_solver_fb.SetSigmaUpdatePolicy(cadence, interval);
      igr_solver_fb.SetStageRtolScale(1.0);

      mfem::EulerFlux base_flux_fb(dim_s, gamma);
      mfem::FluxWithIGR igr_flux_fb(base_flux_fb);
      igr_flux_fb.SetSigmaField(&igr_solver_fb.GetSigma());
      mfem::RusanovFluxWithSides num_flux_fb(igr_flux_fb);

      auto form_integrator_fb =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux_fb,
                                                         /*IntOrderOffset=*/2);
      mfem::DGHyperbolicConservationLaws op_fb(state_fes_s,
                                               std::move(form_integrator_fb),
                                               /*preassembleWeakDivergence=*/true);
      op_fb.SetNumericalFlux(&num_flux_fb);

      mfem::EulerWithIGR wrapped_fb(op_fb, igr_solver_fb, U_base);
      mfem::Vector rhs_fb(U_stage.Size());

      // Intentionally do not call BeginRKStep()/SetRKStage() to validate
      // non-stage-aware fallback behavior.
      wrapped_fb.Mult(U_base, rhs_fb);
      mfem::Vector sigma_fb_base(igr_solver_fb.GetSigma().Size());
      sigma_fb_base = igr_solver_fb.GetSigma();

      wrapped_fb.Mult(U_stage, rhs_fb);
      mfem::Vector sigma_fb_stage(igr_solver_fb.GetSigma().Size());
      sigma_fb_stage = igr_solver_fb.GetSigma();

      igr_solver_fb.SetRKStageIndex(-1);
      igr_solver_fb.UpdateFromState(U_stage, /*warm_start=*/false);
      mfem::Vector sigma_fb_ref(igr_solver_fb.GetSigma().Size());
      sigma_fb_ref = igr_solver_fb.GetSigma();

      mfem::Vector delta_fb_stage_base(sigma_fb_stage);
      delta_fb_stage_base -= sigma_fb_base;
      const double fb_stage_vs_base = GlobalL2Norm(delta_fb_stage_base);

      mfem::Vector delta_fb_ref(sigma_fb_stage);
      delta_fb_ref -= sigma_fb_ref;
      const double fb_stage_err = GlobalL2Norm(delta_fb_ref);

      ctx.LogMetric("igr_non_stage_fallback label=" + label +
                    " stage_err=" + FormatSci(fb_stage_err) +
                    " stage_vs_base=" + FormatSci(fb_stage_vs_base));
      ctx.AddParity(fb_stage_err + fb_stage_vs_base);

      ctx.Check(fb_stage_vs_base > 1.0e-8,
                "EulerWithIGR non-stage fallback updates sigma between distinct states (" +
                  label + ")");
      ctx.Check(fb_stage_err <= 1.0e-5 * fb_stage_vs_base,
                "EulerWithIGR non-stage fallback sigma matches direct stage update (" +
                  label + ")");
    };

    check_non_stage_fallback(igrlab::IGRSigmaUpdateCadence::EveryStep,
                             /*interval=*/1,
                             "every-step");
    check_non_stage_fallback(igrlab::IGRSigmaUpdateCadence::EveryNSteps,
                             /*interval=*/3,
                             "every-n-steps");

    // ExtrapolatedEveryStep is an endpoint-projected index-one DAE method:
    // accepted endpoint closures are exact, while every RHS stage evaluates
    // sigma_n + ((t_stage-t_n)/h_{n-1}) (sigma_n-sigma_{n-1}).  In particular,
    // the prediction must follow physical stage time rather than stage index,
    // because SSPRK(10,4) has non-monotone abscissae c4=2/3, c5=1/3.
    {
      igrlab::IGRSolver extrapolated_solver(sigma_fes_s,
                                            state_fes_s,
                                            dim_s,
                                            num_eq_s,
                                            /*alpha=*/0.25);
      configure_checkpoint_solver(extrapolated_solver);
      extrapolated_solver.SetSigmaUpdatePolicy(
        igrlab::IGRSigmaUpdateCadence::ExtrapolatedEveryStep,
        /*interval=*/1);
      extrapolated_solver.SetStageRtolScale(1.0);

      mfem::EulerFlux extrapolated_base_flux(dim_s, gamma);
      mfem::FluxWithIGR extrapolated_igr_flux(extrapolated_base_flux);
      extrapolated_igr_flux.SetSigmaField(&extrapolated_solver.GetSigma());
      extrapolated_igr_flux.SetFrozenSigmaSpectralSpeed(true);
      mfem::RusanovFluxWithSides extrapolated_num_flux(extrapolated_igr_flux);
      auto extrapolated_form =
        std::make_unique<mfem::HyperbolicFormIntegrator>(
          extrapolated_num_flux,
          /*IntOrderOffset=*/2);
      mfem::DGHyperbolicConservationLaws extrapolated_op(
        state_fes_s,
        std::move(extrapolated_form),
        /*preassembleWeakDivergence=*/true);
      extrapolated_op.SetNumericalFlux(&extrapolated_num_flux);
      mfem::ParGridFunction extrapolated_state(&state_fes_s);
      extrapolated_state = U_base;
      extrapolated_state.ExchangeFaceNbrData();
      mfem::EulerWithIGR extrapolated_wrapped(
        extrapolated_op, extrapolated_solver, extrapolated_state);
      mfem::Vector extrapolated_rhs(extrapolated_state.Size());

      constexpr std::uint64_t generation0 = 400;
      constexpr double h0 = 0.01;
      const mfem::IGRSigmaPreparationResult initial_anchor =
        extrapolated_wrapped.PrepareSigmaForOutput(
          extrapolated_state,
          generation0,
          /*next_rk_step=*/0,
          /*time=*/0.0,
          /*reusable_stage0=*/true);
      mfem::Vector sigma0(extrapolated_solver.GetSigma());
      const int solves_after_initial_anchor =
        extrapolated_solver.GetLinearSolveCalls();
      const long long certificates_after_initial_anchor =
        extrapolated_wrapped.GetSigmaExactCertificateCount();

      // RunTimeLoop performs this direct RHS probe before BeginRKStep when CFL
      // control is active. It must preserve the exact initial anchor and its
      // stage-0 prefetch transaction.
      extrapolated_wrapped.Mult(extrapolated_state, extrapolated_rhs);
      ctx.Check(extrapolated_solver.GetLinearSolveCalls() ==
                    solves_after_initial_anchor &&
                  extrapolated_solver.HasPendingStage0PrefetchAccounting(),
                "IGR extrapolated cadence reuses the exact initial anchor for "
                "the pre-step CFL probe");

      extrapolated_wrapped.BeginRKStep(
        /*step=*/0, /*t=*/0.0, /*dt=*/h0, /*stage_count=*/10);
      extrapolated_wrapped.SetRKStageState(
        /*stage=*/0, generation0, /*stage_time=*/0.0);
      extrapolated_wrapped.Mult(extrapolated_state, extrapolated_rhs);
      mfem::Vector first_stage_error(extrapolated_solver.GetSigma());
      first_stage_error -= sigma0;

      extrapolated_wrapped.SetRKStageState(
        /*stage=*/4,
        generation0 + 1,
        /*stage_time=*/(2.0 / 3.0) * h0);
      extrapolated_wrapped.Mult(U_stage, extrapolated_rhs);
      mfem::Vector startup_hold_error(extrapolated_solver.GetSigma());
      startup_hold_error -= sigma0;

      extrapolated_state = U_stage;
      extrapolated_state.ExchangeFaceNbrData();
      extrapolated_wrapped.EndRKStep(
        extrapolated_state, /*t=*/h0, generation0 + 2);
      mfem::Vector sigma1(extrapolated_solver.GetSigma());
      mfem::Vector first_endpoint_change(sigma1);
      first_endpoint_change -= sigma0;
      const double endpoint_change = GlobalL2Norm(first_endpoint_change);

      ctx.Check(initial_anchor.solved && initial_anchor.reusable_stage0 &&
                  certificates_after_initial_anchor == 1 &&
                  first_stage_error.Normlinf() == 0.0 &&
                  startup_hold_error.Normlinf() == 0.0,
                "IGR extrapolated cadence starts from one exact anchor and "
                "an independent FP64 certificate, then holds it during the "
                "history-free startup step");
      ctx.Check(extrapolated_solver.GetLinearSolveCalls() ==
                    solves_after_initial_anchor + 1 &&
                  extrapolated_wrapped.GetSigmaEndpointSolveCount() == 1 &&
                  extrapolated_wrapped.GetSigmaExtrapolatedStageCount() == 2 &&
                  extrapolated_wrapped.GetSigmaExactCertificateCount() == 2 &&
                  extrapolated_wrapped
                      .GetMaximumLocalSigmaExactCertificateAcceptanceRatio() <=
                    1.0,
                "IGR extrapolated cadence performs no stage solves and one "
                "independently certified post-limiter endpoint solve");
      ctx.Check(endpoint_change > 1.0e-8 &&
                  extrapolated_solver.GetLastLinearConverged() &&
                  extrapolated_wrapped.GetSigmaStateGeneration() ==
                    generation0 + 2,
                "IGR extrapolated cadence commits a distinct exact endpoint "
                "with accepted-state provenance");
      ctx.Check(extrapolated_wrapped.LimitProposedRKStepSize(0.005) == 0.005 &&
                  extrapolated_wrapped.LimitProposedRKStepSize(0.02) == 0.02 &&
                  extrapolated_wrapped.LimitProposedRKStepSize(0.03) == 0.02,
                "IGR extrapolated cadence bounds accepted-step growth at two "
                "without perturbing smaller proposals");

      constexpr double h1 = 0.02;
      const std::uint64_t generation1 = generation0 + 2;
      extrapolated_wrapped.BeginRKStep(
        /*step=*/1, /*t=*/h0, /*dt=*/h1, /*stage_count=*/10);
      extrapolated_wrapped.SetRKStageState(
        /*stage=*/0, generation1, /*stage_time=*/h0);
      extrapolated_wrapped.Mult(extrapolated_state, extrapolated_rhs);
      mfem::Vector second_stage0_error(extrapolated_solver.GetSigma());
      second_stage0_error -= sigma1;

      mfem::Vector sigma_delta(sigma1);
      sigma_delta -= sigma0;
      auto predicted_sigma = [&](double stage_time)
      {
        mfem::Vector prediction(sigma1);
        prediction.Add((stage_time - h0) / h0, sigma_delta);
        return prediction;
      };
      auto prediction_error = [&](double stage_time)
      {
        mfem::Vector error(extrapolated_solver.GetSigma());
        error -= predicted_sigma(stage_time);
        return error.Normlinf();
      };

      const int solves_before_predicted_stages =
        extrapolated_solver.GetLinearSolveCalls();
      const double stage4_time = h0 + (2.0 / 3.0) * h1;
      extrapolated_wrapped.SetRKStageState(
        /*stage=*/4, generation1 + 1, stage4_time);
      extrapolated_wrapped.Mult(U_base, extrapolated_rhs);
      const double stage4_prediction_error = prediction_error(stage4_time);

      const double stage5_time = h0 + (1.0 / 3.0) * h1;
      extrapolated_wrapped.SetRKStageState(
        /*stage=*/5, generation1 + 2, stage5_time);
      extrapolated_wrapped.Mult(U_stage, extrapolated_rhs);
      const double stage5_prediction_error = prediction_error(stage5_time);
      const double prediction_scale =
        std::max({1.0, sigma0.Normlinf(), sigma1.Normlinf()});

      ctx.Check(second_stage0_error.Normlinf() == 0.0 &&
                  stage4_prediction_error <= 32.0 *
                    std::numeric_limits<double>::epsilon() * prediction_scale &&
                  stage5_prediction_error <= 32.0 *
                    std::numeric_limits<double>::epsilon() * prediction_scale,
                "IGR extrapolated cadence maps variable dt and non-monotone "
                "SSPRK(10,4) abscissae to physical-time predictions");
      ctx.Check(extrapolated_solver.GetLinearSolveCalls() ==
                  solves_before_predicted_stages,
                "IGR extrapolated cadence adds no closure solve at predicted "
                "RHS stages");

      // A rejected attempt must restore the exact accepted anchor and leave
      // both endpoint history and solve counts unchanged.  The retried stage
      // then uses its new dt with the same accepted sigma difference.
      extrapolated_wrapped.AbortRKStep(extrapolated_state,
                                       /*restored_time=*/h0,
                                       generation1,
                                       /*retry_dt=*/0.005);
      mfem::Vector rollback_error(extrapolated_solver.GetSigma());
      rollback_error -= sigma1;
      ctx.Check(rollback_error.Normlinf() == 0.0 &&
                  extrapolated_wrapped.GetSigmaStateGeneration() == generation1 &&
                  extrapolated_solver.GetLinearSolveCalls() ==
                    solves_before_predicted_stages &&
                  extrapolated_wrapped.GetSigmaEndpointSolveCount() == 1,
                "IGR extrapolated cadence rollback restores the immutable "
                "accepted anchor without a closure solve");

      constexpr double retry_dt = 0.005;
      extrapolated_wrapped.BeginRKStep(
        /*step=*/1, /*t=*/h0, /*dt=*/retry_dt, /*stage_count=*/10);
      extrapolated_wrapped.SetRKStageState(
        /*stage=*/0, generation1, /*stage_time=*/h0);
      extrapolated_wrapped.Mult(extrapolated_state, extrapolated_rhs);
      const double retry_stage4_time = h0 + (2.0 / 3.0) * retry_dt;
      extrapolated_wrapped.SetRKStageState(
        /*stage=*/4, generation1 + 1, retry_stage4_time);
      extrapolated_wrapped.Mult(U_base, extrapolated_rhs);
      const double retry_prediction_error =
        prediction_error(retry_stage4_time);
      ctx.Check(retry_prediction_error <= 32.0 *
                  std::numeric_limits<double>::epsilon() * prediction_scale,
                "IGR extrapolated cadence retry uses the reduced dt without "
                "mutating accepted history");

      extrapolated_state = U_base;
      extrapolated_state.ExchangeFaceNbrData();
      extrapolated_wrapped.EndRKStep(extrapolated_state,
                                     /*t=*/h0 + retry_dt,
                                     generation1 + 2);
      const double endpoint_defect =
        extrapolated_wrapped.GetMaximumLocalSigmaEndpointPredictionDefect();
      const double relative_endpoint_defect = extrapolated_wrapped
        .GetMaximumLocalRelativeSigmaEndpointPredictionDefect();
      ctx.LogMetric("igr_extrapolated_every_step endpoint_solves=" +
                    std::to_string(
                      extrapolated_wrapped.GetSigmaEndpointSolveCount()) +
                    " stage_predictions=" +
                    std::to_string(
                      extrapolated_wrapped.GetSigmaExtrapolatedStageCount()) +
                    " max_endpoint_defect=" + FormatSci(endpoint_defect) +
                    " max_relative_endpoint_defect=" +
                    FormatSci(relative_endpoint_defect));
      ctx.AddParity(endpoint_defect + relative_endpoint_defect);
      ctx.Check(extrapolated_wrapped.GetSigmaEndpointSolveCount() == 2 &&
                  extrapolated_solver.GetLinearSolveCalls() ==
                    solves_before_predicted_stages + 1 &&
                  std::isfinite(endpoint_defect) && endpoint_defect > 0.0 &&
                  std::isfinite(relative_endpoint_defect),
                "IGR extrapolated cadence certifies each accepted endpoint and "
                "reports a finite prediction defect");

      const long long stage_predictions_before_ssprk104 =
        extrapolated_wrapped.GetSigmaExtrapolatedStageCount();
      const long long endpoint_solves_before_ssprk104 =
        extrapolated_wrapped.GetSigmaEndpointSolveCount();
      const int linear_solves_before_ssprk104 =
        extrapolated_solver.GetLinearSolveCalls();
      mfem::RK3SSPWithLimiter extrapolated_ssprk104(
        state_fes_s, mfem::SSPRKMethod::SSPRK104);
      extrapolated_ssprk104.SetEulerParams(dim_s, gamma);
      extrapolated_ssprk104.SetLimiterParams(
        /*rho_min=*/1.0e-10,
        /*p_min=*/1.0e-10,
        /*quad_order=*/6,
        /*p_bisect_iters=*/12);
      extrapolated_ssprk104.LimitInitial(true);
      extrapolated_ssprk104.AddZhangShuLimiter(
        mfem::ZhangShuPointSet::kConsumer);
      extrapolated_ssprk104.Init(extrapolated_wrapped);
      extrapolated_ssprk104.RestoreStateGeneration(generation1 + 2);
      extrapolated_ssprk104.SetStepCount(/*step=*/2);
      mfem::real_t integrated_time = h0 + retry_dt;
      mfem::real_t integrated_dt = 1.0e-8;
      extrapolated_ssprk104.Step(
        extrapolated_state, integrated_time, integrated_dt);
      ctx.Check(extrapolated_wrapped.GetSigmaExtrapolatedStageCount() ==
                    stage_predictions_before_ssprk104 + 10 &&
                  extrapolated_wrapped.GetSigmaEndpointSolveCount() ==
                    endpoint_solves_before_ssprk104 + 1 &&
                  extrapolated_solver.GetLinearSolveCalls() ==
                    linear_solves_before_ssprk104 + 1,
                "SSPRK(10,4) drives ten extrapolated sigma stages and one "
                "exact accepted endpoint closure while a no-op entry limiter "
                "preserves the previous endpoint provenance");
      ctx.CheckNear(integrated_time,
                    h0 + retry_dt + integrated_dt,
                    10.0 * std::numeric_limits<mfem::real_t>::epsilon(),
                    "SSPRK(10,4) advances the extrapolated DAE candidate");
    }

    // Extrapolated-stage q_sigma admission uses only samples contributed by
    // the just-completed RHS. A rejection restores the exact accepted anchor;
    // its cumulative telemetry must not reject the subsequent retry again.
    {
      igrlab::IGRSolver admission_solver(sigma_fes_s,
                                         state_fes_s,
                                         dim_s,
                                         num_eq_s,
                                         /*alpha=*/0.25);
      configure_checkpoint_solver(admission_solver);
      admission_solver.SetSigmaUpdatePolicy(
        igrlab::IGRSigmaUpdateCadence::ExtrapolatedEveryStep,
        /*interval=*/1);
      admission_solver.SetStageRtolScale(1.0);

      mfem::EulerFlux admission_base_flux(dim_s, gamma);
      mfem::FluxWithIGR admission_igr_flux(admission_base_flux);
      admission_igr_flux.SetSigmaField(&admission_solver.GetSigma());
      mfem::RusanovFluxWithSides admission_num_flux(admission_igr_flux);
      auto admission_form =
        std::make_unique<mfem::HyperbolicFormIntegrator>(
          admission_num_flux,
          /*IntOrderOffset=*/2);
      SyntheticQSigmaTelemetryDG admission_op(
        state_fes_s,
        std::move(admission_form),
        /*preassembleWeakDivergence=*/true);
      admission_op.SetNumericalFlux(&admission_num_flux);
      mfem::ParGridFunction admission_state(&state_fes_s);
      admission_state = U_base;
      admission_state.ExchangeFaceNbrData();
      mfem::EulerWithIGR admission_wrapped(
        admission_op, admission_solver, admission_state);
      mfem::Vector admission_rhs(admission_state.Size());

      constexpr std::uint64_t admission_generation = 700;
      constexpr double admission_dt = 0.01;
      admission_wrapped.PrepareSigmaForOutput(
        admission_state,
        admission_generation,
        /*next_rk_step=*/0,
        /*time=*/0.0,
        /*reusable_stage0=*/true);
      const mfem::Vector accepted_sigma(admission_solver.GetSigma());

      admission_op.SeedHistoricalFailure(/*nonpositive=*/11,
                                         /*nonfinite=*/7);
      admission_wrapped.BeginRKStep(
        /*step=*/0,
        /*t=*/0.0,
        /*dt=*/admission_dt,
        /*stage_count=*/10);
      admission_wrapped.SetRKStageState(
        /*stage=*/0,
        admission_generation,
        /*stage_time=*/0.0);
      bool historical_failure_replayed = false;
      try
      {
        admission_wrapped.Mult(admission_state, admission_rhs);
      }
      catch (const mfem::RecoverableRKStageAdmissionFailure&)
      {
        historical_failure_replayed = true;
      }

      admission_op.FailNextRHS(/*nonpositive=*/3, /*nonfinite=*/2);
      admission_wrapped.SetRKStageState(
        /*stage=*/4,
        admission_generation + 1,
        /*stage_time=*/(2.0 / 3.0) * admission_dt);
      bool current_rhs_rejected = false;
      try
      {
        admission_wrapped.Mult(admission_state, admission_rhs);
      }
      catch (const mfem::RecoverableRKStageAdmissionFailure& failure)
      {
        current_rhs_rejected =
          failure.Reason() == mfem::RKStageAdmissionFailureReason::
                                QSigmaNonpositiveAndNonfinite &&
          failure.GlobalNonpositiveQSigmaSampleCount() == 3 &&
          failure.GlobalNonfiniteQSigmaSampleCount() == 2 &&
          failure.Stage() == 4;
      }
      admission_wrapped.AbortRKStep(admission_state,
                                    /*restored_time=*/0.0,
                                    admission_generation,
                                    /*retry_dt=*/0.005);
      mfem::Vector admission_rollback_error(admission_solver.GetSigma());
      admission_rollback_error -= accepted_sigma;

      admission_wrapped.BeginRKStep(
        /*step=*/0,
        /*t=*/0.0,
        /*dt=*/0.005,
        /*stage_count=*/10);
      admission_wrapped.SetRKStageState(
        /*stage=*/0,
        admission_generation,
        /*stage_time=*/0.0);
      bool retry_replayed_failure = false;
      try
      {
        admission_wrapped.Mult(admission_state, admission_rhs);
      }
      catch (const mfem::RecoverableRKStageAdmissionFailure&)
      {
        retry_replayed_failure = true;
      }
      admission_wrapped.AbortRKStep(admission_state,
                                    /*restored_time=*/0.0,
                                    admission_generation,
                                    /*retry_dt=*/0.0);

      // Stage samples can all be admissible while the exact closure
      // reconstructed after the terminal limiter is not. That endpoint must
      // reject before committing sigma history, and the producing step must
      // remain rollback-safe.
      admission_wrapped.BeginRKStep(
        /*step=*/0,
        /*t=*/0.0,
        /*dt=*/admission_dt,
        /*stage_count=*/10);
      admission_wrapped.SetRKStageState(
        /*stage=*/0,
        admission_generation,
        /*stage_time=*/0.0);
      admission_wrapped.Mult(admission_state, admission_rhs);
      admission_op.FailNextEndpointAdmission(/*nonpositive=*/4,
                                             /*nonfinite=*/1);
      bool exact_endpoint_rejected = false;
      try
      {
        admission_wrapped.EndRKStep(admission_state,
                                    /*t=*/admission_dt,
                                    admission_generation + 1);
      }
      catch (const mfem::RecoverableRKStageAdmissionFailure& failure)
      {
        exact_endpoint_rejected =
          failure.Reason() == mfem::RKStageAdmissionFailureReason::
                                QSigmaNonpositiveAndNonfinite &&
          failure.GlobalNonpositiveQSigmaSampleCount() == 4 &&
          failure.GlobalNonfiniteQSigmaSampleCount() == 1 &&
          failure.Stage() == -1;
      }
      admission_wrapped.AbortRKStep(admission_state,
                                    /*restored_time=*/0.0,
                                    admission_generation,
                                    /*retry_dt=*/0.005);
      mfem::Vector endpoint_rollback_error(admission_solver.GetSigma());
      endpoint_rollback_error -= accepted_sigma;

      ctx.Check(admission_wrapped.RequiresRKStepTransaction(),
                "IGR extrapolated q_sigma admission requests an RK rollback "
                "transaction");
      ctx.Check(!historical_failure_replayed && current_rhs_rejected &&
                  !retry_replayed_failure,
                "IGR extrapolated q_sigma admission rejects only violations "
                "newly observed by the current RHS");
      ctx.Check(admission_rollback_error.Normlinf() == 0.0 &&
                  exact_endpoint_rejected &&
                  endpoint_rollback_error.Normlinf() == 0.0 &&
                  admission_wrapped.GetSigmaStateGeneration() ==
                    admission_generation,
                "IGR extrapolated stage and exact-endpoint q_sigma rejection "
                "restore the immutable accepted sigma anchor");
      ctx.Check(admission_wrapped.GetSigmaStageAdmissionRejectionCount() == 2 &&
                  admission_wrapped
                      .GetSigmaStageAdmissionNonpositiveSampleCount() == 7 &&
                  admission_wrapped
                      .GetSigmaStageAdmissionNonfiniteSampleCount() == 3 &&
                  std::string(admission_wrapped
                                .GetLastSigmaStageAdmissionFailureReason()) ==
                    "q-sigma-nonpositive-and-nonfinite",
                "IGR extrapolated stage and exact-endpoint q_sigma admission "
                "report bounded attempted-work counts and the last reason");
    }

    // --- Rejected RK attempts restore IGR cadence and stage-0 provenance ---
    auto check_refresh_rejection =
      [&](igrlab::IGRSigmaUpdateCadence cadence,
          bool expect_abort_solve,
          const std::string& label)
    {
      igrlab::IGRSolver retry_solver(sigma_fes_s,
                                     state_fes_s,
                                     dim_s,
                                     num_eq_s,
                                     /*alpha=*/0.25);
      configure_checkpoint_solver(retry_solver);
      retry_solver.SetSigmaUpdatePolicy(cadence, /*interval=*/1);
      retry_solver.SetStageRtolScale(1.0);

      mfem::EulerFlux retry_base_flux(dim_s, gamma);
      mfem::FluxWithIGR retry_igr_flux(retry_base_flux);
      retry_igr_flux.SetSigmaField(&retry_solver.GetSigma());
      mfem::RusanovFluxWithSides retry_num_flux(retry_igr_flux);
      auto retry_form =
        std::make_unique<mfem::HyperbolicFormIntegrator>(retry_num_flux,
                                                         /*IntOrderOffset=*/2);
      mfem::DGHyperbolicConservationLaws retry_op(
        state_fes_s, std::move(retry_form), /*preassembleWeakDivergence=*/true);
      retry_op.SetNumericalFlux(&retry_num_flux);
      mfem::EulerWithIGR retry_wrapped(retry_op, retry_solver, U_base);

      constexpr std::uint64_t restored_generation = 200;
      retry_wrapped.PrepareSigmaForOutput(
        U_base,
        restored_generation,
        /*next_rk_step=*/0,
        /*time=*/0.0,
        /*reusable_stage0=*/true);
      mfem::Vector prepared_sigma(retry_solver.GetSigma());
      const double prepared_norm =
        std::max(GlobalL2Norm(prepared_sigma), 1.0e-30);
      const long long reuse_before = retry_wrapped.GetSigmaStage0ReuseCount();

      mfem::Vector retry_rhs(U_base.Size());
      retry_wrapped.BeginRKStep(/*step=*/0, /*t=*/0.0, /*dt=*/0.02);
      retry_wrapped.SetRKStageState(/*stage=*/0, restored_generation);
      retry_wrapped.Mult(U_base, retry_rhs);
      retry_wrapped.SetRKStageState(/*stage=*/1,
                                    restored_generation + 1);
      retry_wrapped.Mult(U_stage, retry_rhs);
      const int calls_before_abort = retry_solver.GetLinearSolveCalls();

      retry_wrapped.AbortRKStep(U_base,
                               /*restored_time=*/0.0,
                               restored_generation,
                               /*retry_dt=*/0.01);
      const int expected_abort_calls = expect_abort_solve ? 1 : 0;
      ctx.Check(retry_solver.GetLinearSolveCalls() ==
                  calls_before_abort + expected_abort_calls,
                "IGR retry lifecycle: abort solve disposition is correct (" +
                  label + ")");
      ctx.Check(retry_wrapped.GetSigmaStateGeneration() ==
                  restored_generation,
                "IGR retry lifecycle: abort restores sigma provenance (" +
                  label + ")");
      ctx.Check(retry_solver.HasPendingStage0PrefetchAccounting() ==
                  expect_abort_solve,
                "IGR retry lifecycle: abort prefetch accounting matches "
                "actual retry preparation (" +
                  label + ")");
      mfem::Vector abort_delta(retry_solver.GetSigma());
      abort_delta -= prepared_sigma;
      ctx.Check(GlobalL2Norm(abort_delta) <= 1.0e-5 * prepared_norm,
                "IGR retry lifecycle: abort sigma matches restored state (" +
                  label + ")");

      const int calls_before_retry = retry_solver.GetLinearSolveCalls();
      retry_wrapped.BeginRKStep(/*step=*/0, /*t=*/0.0, /*dt=*/0.01);
      retry_wrapped.SetRKStageState(/*stage=*/0, restored_generation);
      retry_wrapped.Mult(U_base, retry_rhs);
      ctx.Check(retry_solver.GetLinearSolveCalls() == calls_before_retry,
                "IGR retry lifecycle: retry consumes prepared stage 0 (" +
                  label + ")");
      ctx.Check(!retry_solver.HasPendingStage0PrefetchAccounting(),
                "IGR retry lifecycle: retry leaves no pending prefetch "
                "accounting (" +
                  label + ")");
      ctx.Check(retry_wrapped.GetSigmaStage0ReuseCount() == reuse_before + 2,
                "IGR retry lifecycle: rejected and retried stage 0 are both "
                "accounted (" +
                  label + ")");
      retry_wrapped.EndRKStep(
        U_base, /*t=*/0.01, restored_generation + 2);
    };

    check_refresh_rejection(igrlab::IGRSigmaUpdateCadence::EveryStage,
                            /*expect_abort_solve=*/true,
                            "every-stage");
    check_refresh_rejection(igrlab::IGRSigmaUpdateCadence::EveryStep,
                            /*expect_abort_solve=*/false,
                            "every-step");

    // A skipped EveryNSteps step must remain skipped across rejection. The
    // output immediately before it is temporary and must not change either an
    // accepted or rejected transport attempt. The next scheduled refresh must
    // still be recognized and reusable on retry.
    {
      igrlab::IGRSolver retry_solver(sigma_fes_s,
                                     state_fes_s,
                                     dim_s,
                                     num_eq_s,
                                     /*alpha=*/0.25);
      configure_checkpoint_solver(retry_solver);
      retry_solver.SetSigmaUpdatePolicy(
        igrlab::IGRSigmaUpdateCadence::EveryNSteps, /*interval=*/3);
      retry_solver.SetStageRtolScale(1.0);

      mfem::EulerFlux retry_base_flux(dim_s, gamma);
      mfem::FluxWithIGR retry_igr_flux(retry_base_flux);
      retry_igr_flux.SetSigmaField(&retry_solver.GetSigma());
      mfem::RusanovFluxWithSides retry_num_flux(retry_igr_flux);
      auto retry_form =
        std::make_unique<mfem::HyperbolicFormIntegrator>(retry_num_flux,
                                                         /*IntOrderOffset=*/2);
      mfem::DGHyperbolicConservationLaws retry_op(
        state_fes_s, std::move(retry_form), /*preassembleWeakDivergence=*/true);
      retry_op.SetNumericalFlux(&retry_num_flux);
      mfem::ParGridFunction retry_state(&state_fes_s);
      retry_state = U_base;
      retry_state.ExchangeFaceNbrData();
      mfem::EulerWithIGR retry_wrapped(
        retry_op, retry_solver, retry_state);
      mfem::Vector retry_rhs(retry_state.Size());

      constexpr std::uint64_t generation0 = 300;
      retry_wrapped.PrepareSigmaForOutput(retry_state,
                                          generation0,
                                          /*next_rk_step=*/0,
                                          /*time=*/0.0,
                                          /*reusable_stage0=*/true);
      retry_wrapped.BeginRKStep(/*step=*/0, /*t=*/0.0, /*dt=*/0.01);
      retry_wrapped.SetRKStageState(/*stage=*/0, generation0);
      retry_wrapped.Mult(retry_state, retry_rhs);
      mfem::Vector production_sigma(retry_solver.GetSigma());
      retry_state = U_stage;
      retry_state.ExchangeFaceNbrData();
      retry_wrapped.EndRKStep(retry_state, /*t=*/0.01, generation0 + 1);
      const long long reuse_after_step0 =
        retry_wrapped.GetSigmaStage0ReuseCount();
      const int production_calls_after_step0 =
        retry_solver.GetProductionLinearSolveCalls();

      const mfem::IGRSigmaPreparationResult accepted_skip_output =
        retry_wrapped.PrepareSigmaForOutput(
          retry_state,
          generation0 + 1,
          /*next_rk_step=*/1,
          /*time=*/0.01,
          /*reusable_stage0=*/true);
      ctx.Check(accepted_skip_output.solved &&
                  !accepted_skip_output.reusable_stage0 &&
                  accepted_skip_output.output_only &&
                  retry_solver.GetProductionLinearSolveCalls() ==
                    production_calls_after_step0,
                "IGR output isolation: pre-skip output is not a production "
                "prefetch");
      mfem::Vector accepted_output_delta(retry_solver.GetSigma());
      accepted_output_delta -= production_sigma;
      ctx.Check(GlobalL2Norm(accepted_output_delta) > 1.0e-8,
                "IGR output isolation: accepted-step output setup changes the "
                "visible diagnostic sigma");
      const int calls_after_accepted_output =
        retry_solver.GetLinearSolveCalls();
      const int output_calls_after_accepted_output =
        retry_solver.GetOutputOnlyLinearSolveCalls();

      // Accepted skipped step: BeginRKStep restores production sigma before
      // the RHS and the cadence adds no solve or stage-0 reuse.
      retry_wrapped.BeginRKStep(/*step=*/1, /*t=*/0.01, /*dt=*/0.02);
      retry_wrapped.SetRKStageState(/*stage=*/0, generation0 + 1);
      mfem::Vector restored_before_accepted_rhs(retry_solver.GetSigma());
      restored_before_accepted_rhs -= production_sigma;
      retry_wrapped.Mult(retry_state, retry_rhs);
      ctx.Check(restored_before_accepted_rhs.Normlinf() == 0.0 &&
                  retry_solver.GetLinearSolveCalls() ==
                    calls_after_accepted_output &&
                  retry_wrapped.GetSigmaStage0ReuseCount() ==
                    reuse_after_step0,
                "IGR output isolation: accepted skipped step restores and "
                "retains production sigma");
      retry_wrapped.EndRKStep(
        retry_state, /*t=*/0.03, generation0 + 2);

      // Rejected skipped step: a second immediate output solve is temporary;
      // both the failed attempt and retry retain the same production sigma.
      const mfem::IGRSigmaPreparationResult rejected_skip_output =
        retry_wrapped.PrepareSigmaForOutput(
          retry_state,
          generation0 + 2,
          /*next_rk_step=*/2,
          /*time=*/0.03,
          /*reusable_stage0=*/true);
      ctx.Check(rejected_skip_output.solved &&
                  !rejected_skip_output.reusable_stage0 &&
                  rejected_skip_output.output_only &&
                  retry_solver.GetOutputOnlyLinearSolveCalls() ==
                    output_calls_after_accepted_output + 1 &&
                  retry_solver.GetProductionLinearSolveCalls() ==
                    production_calls_after_step0,
                "IGR output isolation: rejected pre-skip output is separately "
                "classified as output-only");
      const int calls_after_rejected_output =
        retry_solver.GetLinearSolveCalls();
      retry_wrapped.BeginRKStep(/*step=*/2, /*t=*/0.03, /*dt=*/0.02);
      retry_wrapped.SetRKStageState(/*stage=*/0, generation0 + 2);
      retry_wrapped.Mult(retry_state, retry_rhs);
      retry_wrapped.SetRKStageState(/*stage=*/1, generation0 + 3);
      retry_wrapped.Mult(retry_state, retry_rhs);
      retry_wrapped.AbortRKStep(retry_state,
                               /*restored_time=*/0.03,
                               generation0 + 2,
                               /*retry_dt=*/0.01);
      retry_wrapped.BeginRKStep(/*step=*/2, /*t=*/0.03, /*dt=*/0.01);
      retry_wrapped.SetRKStageState(/*stage=*/0, generation0 + 2);
      retry_wrapped.Mult(retry_state, retry_rhs);
      mfem::Vector rejected_retry_sigma_delta(retry_solver.GetSigma());
      rejected_retry_sigma_delta -= production_sigma;
      ctx.Check(rejected_retry_sigma_delta.Normlinf() == 0.0 &&
                  retry_solver.GetLinearSolveCalls() ==
                    calls_after_rejected_output &&
                  retry_wrapped.GetSigmaStage0ReuseCount() ==
                    reuse_after_step0,
                "IGR output isolation: rejected skipped step and retry add no "
                "closure solve or reuse");
      retry_wrapped.EndRKStep(
        retry_state, /*t=*/0.04, generation0 + 4);

      retry_wrapped.BeginRKStep(/*step=*/3, /*t=*/0.04, /*dt=*/0.02);
      retry_wrapped.SetRKStageState(/*stage=*/0, generation0 + 4);
      retry_wrapped.Mult(retry_state, retry_rhs);
      const int calls_after_scheduled_refresh =
        retry_solver.GetLinearSolveCalls();
      retry_wrapped.AbortRKStep(retry_state,
                               /*restored_time=*/0.04,
                               generation0 + 4,
                               /*retry_dt=*/0.01);
      retry_wrapped.BeginRKStep(/*step=*/3, /*t=*/0.04, /*dt=*/0.01);
      retry_wrapped.SetRKStageState(/*stage=*/0, generation0 + 4);
      retry_wrapped.Mult(retry_state, retry_rhs);
      ctx.Check(retry_solver.GetLinearSolveCalls() ==
                  calls_after_scheduled_refresh &&
                  retry_wrapped.GetSigmaStage0ReuseCount() ==
                    reuse_after_step0 + 1,
                "IGR retry lifecycle: scheduled every-N refresh remains due "
                "and is reused on retry");
      retry_wrapped.EndRKStep(
        retry_state, /*t=*/0.05, generation0 + 5);
    }
  }

  // Exercise both the MFEM-matching arithmetic shared-face path and the
  // opt-in harmonic-weighted candidate on an actual partition boundary.
  if (mfem::Mpi::WorldSize() > 1)
  {
    constexpr int dim_shared = 2;
    mfem::Mesh mesh_shared = MakeGradedQuadMesh2D();
    mfem::ParMesh pmesh_shared(MPI_COMM_WORLD, mesh_shared);

    mfem::DG_FECollection shared_fec(/*order=*/1,
                                     dim_shared,
                                     mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace shared_fes(&pmesh_shared,
                                           &shared_fec,
                                           /*vdim=*/1,
                                           mfem::Ordering::byNODES);
    const mfem::IntegrationRule& shared_face_ir =
      mfem::IntRules.Get(pmesh_shared.GetTypicalFaceGeometry(),
                         /*order=*/2);
    mfem::FaceQuadratureSpace shared_fqs(pmesh_shared,
                                         shared_face_ir,
                                         mfem::FaceType::Interior);

    int local_shared_in_fqs = 0;
    for (int f = 0; f < shared_fqs.GetNE(); ++f)
    {
      const int mesh_face = shared_fqs.GetMeshFaceIndex(f);
      const mfem::Mesh::FaceInformation face =
        pmesh_shared.GetFaceInformation(mesh_face);
      if (face.IsShared())
      {
        local_shared_in_fqs++;
      }
    }

    int global_shared_in_fqs = 0;
    int global_pmesh_shared = 0;
    const int local_pmesh_shared = pmesh_shared.GetNSharedFaces();
    MPI_Allreduce(&local_shared_in_fqs,
                  &global_shared_in_fqs,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_pmesh_shared,
                  &global_pmesh_shared,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    std::string side_reason;
    const bool side_supported =
      igrlab::dg_face::SupportsInteriorFaceSideDensityProjection2D(
        shared_fes,
        shared_fqs,
        &side_reason);
    std::string trace_reason;
    const bool trace_supported =
      igrlab::dg_face::SupportsInteriorFaceScalarTraceProjection2D(
        shared_fes,
        shared_fqs,
        &trace_reason);

    std::string shared_apply_reason;
    const bool shared_apply_supported =
      igrlab::dg_face::SupportsSharedFaceDGDiffusionApplyData2D(
        shared_fes,
        shared_face_ir,
        &shared_apply_reason);

    int local_shared_alpha_bad = 0;
    double local_shared_alpha_error = 0.0;
    double local_shared_alpha_jump = 0.0;
    double local_shared_alpha_vs_max = 0.0;
    if (local_pmesh_shared > 0)
    {
      constexpr double shared_c_alpha = 0.39;
      constexpr double shared_rho = 1.3;
      constexpr double shared_rho_floor = 0.2;
      mfem::ParGridFunction shared_density(&shared_fes);
      shared_density = shared_rho;
      shared_density.ExchangeFaceNbrData();

      double local_h2_max = 0.0;
      for (int e = 0; e < pmesh_shared.GetNE(); ++e)
      {
        const double h = pmesh_shared.GetElementSize(e, 1);
        local_h2_max = std::max(local_h2_max, h * h);
      }
      double global_h2_max = 0.0;
      MPI_Allreduce(&local_h2_max,
                    &global_h2_max,
                    1,
                    MPI_DOUBLE,
                    MPI_MAX,
                    MPI_COMM_WORLD);

      mfem::Vector face_side_alpha(2 * local_pmesh_shared);
      auto requested_alpha =
        mfem::Reshape(face_side_alpha.HostWrite(), 2, local_pmesh_shared);
      for (int sf = 0; sf < local_pmesh_shared; ++sf)
      {
        mfem::FaceElementTransformations* Tr =
          pmesh_shared.GetSharedFaceTransformations(sf, /*fill2=*/true);
        MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                    "Shared local-alpha projection test needs two elements.");
        const int nbr_el = Tr->Elem2No - pmesh_shared.GetNE();
        MFEM_VERIFY(Tr->Elem1No >= 0 &&
                      Tr->Elem1No < pmesh_shared.GetNE() && nbr_el >= 0,
                    "Shared local-alpha projection test element mismatch.");
        const double h0 = pmesh_shared.GetElementSize(Tr->Elem1No, 1);
        const double h1 = pmesh_shared.GetFaceNbrElementSize(nbr_el, 1);
        requested_alpha(0, sf) = shared_c_alpha * h0 * h0;
        requested_alpha(1, sf) = shared_c_alpha * h1 * h1;
        local_shared_alpha_jump =
          std::max(local_shared_alpha_jump,
                   std::abs(requested_alpha(0, sf) -
                            requested_alpha(1, sf)));
      }

      mfem::Vector projected_local_q;
      mfem::Vector projected_max_q;
      igrlab::dg_face::SharedFaceSideDensityProjectionScratch
        projected_local_scratch;
      igrlab::dg_face::SharedFaceSideDensityProjectionScratch
        projected_max_scratch;
      const auto projected_local_info =
        igrlab::dg_face::ProjectSharedFaceDensityCoefficient2D(
          shared_fes,
          shared_face_ir,
          shared_density,
          face_side_alpha,
          shared_rho_floor,
          projected_local_q,
          &projected_local_scratch);
      const auto projected_max_info =
        igrlab::dg_face::ProjectSharedFaceDensityCoefficient2D(
          shared_fes,
          shared_face_ir,
          shared_density,
          shared_c_alpha * global_h2_max,
          shared_rho_floor,
          projected_max_q,
          &projected_max_scratch);
      if (projected_local_info.nf != local_pmesh_shared ||
          projected_local_info.nf != projected_max_info.nf ||
          projected_local_info.nq != projected_max_info.nq ||
          projected_local_scratch.side_alpha.Size() !=
            2 * local_pmesh_shared ||
          projected_max_scratch.side_alpha.Size() != 0)
      {
        local_shared_alpha_bad = 1;
      }
      else
      {
        const auto requested =
          mfem::Reshape(face_side_alpha.HostRead(), 2, local_pmesh_shared);
        const auto retained =
          mfem::Reshape(projected_local_scratch.side_alpha.HostRead(),
                        2,
                        local_pmesh_shared);
        const auto local_q = mfem::Reshape(projected_local_q.HostRead(),
                                           2,
                                           projected_local_info.nq,
                                           projected_local_info.nf);
        const auto max_q = mfem::Reshape(projected_max_q.HostRead(),
                                         2,
                                         projected_max_info.nq,
                                         projected_max_info.nf);
        for (int f = 0; f < projected_local_info.nf; ++f)
        {
          for (int side = 0; side < 2; ++side)
          {
            local_shared_alpha_error =
              std::max(local_shared_alpha_error,
                       std::abs(retained(side, f) - requested(side, f)));
            for (int q = 0; q < projected_local_info.nq; ++q)
            {
              const double expected = requested(side, f) / shared_rho;
              const double expected_max =
                shared_c_alpha * global_h2_max / shared_rho;
              local_shared_alpha_error =
                std::max({local_shared_alpha_error,
                          std::abs(local_q(side, q, f) - expected),
                          std::abs(max_q(side, q, f) - expected_max)});
              local_shared_alpha_vs_max =
                std::max(local_shared_alpha_vs_max,
                         std::abs(local_q(side, q, f) - max_q(side, q, f)));
            }
          }
        }
        if (local_shared_alpha_error > 5.0e-13)
        {
          local_shared_alpha_bad = 1;
        }
      }
    }

    int local_shared_apply_bad = 0;
    int local_shared_repeat_bad = 0;
    int local_shared_apply_nf = 0;
    if (local_pmesh_shared > 0)
    {
      if (!shared_apply_supported)
      {
        local_shared_apply_bad = 1;
      }
      else
      {
        mfem::Vector shared_side_q(2 *
                                   shared_face_ir.GetNPoints() *
                                   local_pmesh_shared);
        shared_side_q = 1.0;
        igrlab::dg_face::SharedFaceDGDiffusionApplyScratch shared_scratch;
        const auto shared_info =
          igrlab::dg_face::BuildSharedFaceDGDiffusionApplyData2D(
            shared_fes,
            shared_face_ir,
            shared_side_q,
            /*kappa=*/1.0,
            shared_scratch.shape,
            shared_scratch.weighted_normal_dshape,
            shared_scratch.penalty,
            shared_scratch.face_elements,
            shared_scratch.element_vdof_map,
            shared_scratch.face_nbr_dof_map);
        local_shared_apply_nf = shared_info.nf;
        if (shared_info.nf != local_pmesh_shared ||
            shared_info.nq != shared_face_ir.GetNPoints() ||
            shared_scratch.shape.Size() != shared_info.shape_size ||
            shared_scratch.weighted_normal_dshape.Size() !=
              shared_info.weighted_normal_dshape_size ||
            shared_scratch.penalty.Size() != shared_info.penalty_size ||
            shared_scratch.face_elements.Size() !=
              shared_info.face_element_size ||
            shared_scratch.face_nbr_dof_map.Size() !=
              shared_info.face_nbr_dof_map_size)
        {
          local_shared_apply_bad = 1;
        }

        const auto face_el = shared_scratch.face_elements.HostRead();
        for (int sf = 0; sf < shared_scratch.face_elements.Size(); ++sf)
        {
          if (face_el[sf] < 0 || face_el[sf] >= shared_fes.GetNE())
          {
            local_shared_apply_bad = 1;
          }
        }

        const auto nbr_dofs = shared_scratch.face_nbr_dof_map.HostRead();
        for (int i = 0; i < shared_scratch.face_nbr_dof_map.Size(); ++i)
        {
          int vd = nbr_dofs[i];
          if (vd < 0) { vd = -1 - vd; }
          if (vd < 0 || vd >= shared_fes.GetFaceNbrVSize())
          {
            local_shared_apply_bad = 1;
          }
        }

        const auto penalty = shared_scratch.penalty.HostRead();
        for (int i = 0; i < shared_scratch.penalty.Size(); ++i)
        {
          if (!std::isfinite(penalty[i]))
          {
            local_shared_apply_bad = 1;
          }
        }

        const auto ndshape =
          shared_scratch.weighted_normal_dshape.HostRead();
        for (int i = 0;
             i < shared_scratch.weighted_normal_dshape.Size();
             ++i)
        {
          if (!std::isfinite(ndshape[i]))
          {
            local_shared_apply_bad = 1;
          }
        }

        mfem::ParGridFunction shared_sigma(&shared_fes);
        shared_sigma = 0.0;
        mfem::Array<int> shared_dofs;
        const int shared_dof = shared_fes.GetFE(0)->GetDof();
        for (int e = 0; e < shared_fes.GetNE(); ++e)
        {
          shared_fes.GetElementDofs(e, shared_dofs);
          MFEM_VERIFY(shared_dofs.Size() == shared_dof,
                      "Shared exact-face apply test dof mismatch.");
          const double rank_offset =
            0.13 * static_cast<double>(mfem::Mpi::WorldRank());
          for (int d = 0; d < shared_dof; ++d)
          {
            int vdof = shared_fes.DofToVDof(shared_dofs[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            shared_sigma(vdof) =
              0.31 + rank_offset + 0.17 * static_cast<double>(e) -
              0.041 * static_cast<double>(d + 1);
          }
        }

        const double shared_ip_sigma =
          static_cast<double>(igrlab::IGRIPKind::SIPG);
        const double shared_kappa = 2.75;
        mfem::Vector shared_apply_y(shared_sigma.Size());
        shared_apply_y = 0.0;
        igrlab::dg_face::SharedFaceDGDiffusionApplyScratch
          shared_apply_scratch;
        const auto shared_apply_info =
          igrlab::dg_face::AddSharedFaceDGDiffusionApply2D(
            shared_fes,
            shared_face_ir,
            shared_sigma,
            shared_side_q,
            shared_ip_sigma,
            shared_kappa,
            shared_apply_y,
            &shared_apply_scratch);

        mfem::Vector shared_repeat_y0(shared_sigma.Size());
        mfem::Vector shared_repeat_y1(shared_sigma.Size());
        shared_repeat_y0 = 0.0;
        shared_repeat_y1 = 0.0;
        igrlab::dg_face::LaunchSharedFaceDGDiffusionApply2D(
          shared_apply_info,
          shared_apply_scratch,
          shared_sigma,
          shared_sigma.FaceNbrData(),
          shared_ip_sigma,
          shared_repeat_y0);
        igrlab::dg_face::LaunchSharedFaceDGDiffusionApply2D(
          shared_apply_info,
          shared_apply_scratch,
          shared_sigma,
          shared_sigma.FaceNbrData(),
          shared_ip_sigma,
          shared_repeat_y1);
        if (!SameVectorBits(shared_repeat_y0, shared_repeat_y1))
        {
          local_shared_repeat_bad = 1;
        }
        const std::uint64_t shared_first_revision =
          shared_apply_scratch.incidence_connectivity_revision;
        const auto shared_rebuilt_info =
          igrlab::dg_face::BuildSharedFaceDGDiffusionApplyData2D(
            shared_fes,
            shared_face_ir,
            shared_side_q,
            shared_kappa,
            shared_apply_scratch.shape,
            shared_apply_scratch.weighted_normal_dshape,
            shared_apply_scratch.penalty,
            shared_apply_scratch.face_elements,
            shared_apply_scratch.element_vdof_map,
            shared_apply_scratch.face_nbr_dof_map);
        const bool shared_revision_invalidated =
          shared_first_revision == shared_apply_info.connectivity_revision &&
          shared_rebuilt_info.connectivity_revision != shared_first_revision &&
          shared_apply_scratch.incidence_connectivity_revision ==
            shared_first_revision;
        mfem::Vector shared_rebuilt_y(shared_sigma.Size());
        shared_rebuilt_y = 0.0;
        igrlab::dg_face::LaunchSharedFaceDGDiffusionApply2D(
          shared_rebuilt_info,
          shared_apply_scratch,
          shared_sigma,
          shared_sigma.FaceNbrData(),
          shared_ip_sigma,
          shared_rebuilt_y);
        if (!shared_revision_invalidated ||
            shared_apply_scratch.incidence_connectivity_revision !=
              shared_rebuilt_info.connectivity_revision ||
            !SameVectorBits(shared_repeat_y0, shared_rebuilt_y))
        {
          local_shared_repeat_bad = 1;
        }

        mfem::Vector shared_diag(shared_sigma.Size());
        shared_diag = 0.0;
        igrlab::dg_face::AddSharedFaceDGDiffusionLocalDiagonal2D(
          shared_apply_info,
          shared_apply_scratch,
          shared_ip_sigma,
          shared_diag);

        mfem::ConstantCoefficient one(1.0);
        mfem::DGDiffusionIntegrator mfem_shared_face(one,
                                                     shared_ip_sigma,
                                                     shared_kappa);
        mfem_shared_face.SetIntegrationRule(shared_face_ir);
        mfem::Vector assembled_shared_y(shared_sigma.Size());
        assembled_shared_y = 0.0;
        mfem::Vector assembled_shared_diag(shared_sigma.Size());
        assembled_shared_diag = 0.0;
        const double* sigma_data = shared_sigma.HostRead();
        const double* sigma_nbr_data =
          shared_sigma.FaceNbrData().HostRead();
        double* assembled_data = assembled_shared_y.HostReadWrite();
        double* assembled_shared_diag_data =
          assembled_shared_diag.HostReadWrite();
        mfem::DenseMatrix elmat;
        mfem::Vector xloc(2 * shared_dof);
        mfem::Vector yloc(2 * shared_dof);
        mfem::Array<int> nbr_dofs_ref;

        for (int sf = 0; sf < shared_apply_info.nf; ++sf)
        {
          mfem::FaceElementTransformations* Tr =
            pmesh_shared.GetSharedFaceTransformations(sf, /*fill2=*/true);
          MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                      "Shared exact-face apply test needs two elements.");
          const int e1 = Tr->Elem1No;
          const int nbr_el = Tr->Elem2No - pmesh_shared.GetNE();
          MFEM_VERIFY(nbr_el >= 0,
                      "Shared exact-face apply test neighbor mismatch.");

          const mfem::FiniteElement& fe1 = *shared_fes.GetFE(e1);
          const mfem::FiniteElement* fe2_ptr =
            shared_fes.GetFaceNbrFE(nbr_el);
          MFEM_VERIFY(fe2_ptr != nullptr,
                      "Shared exact-face apply test missing neighbor FE.");
          mfem_shared_face.AssembleFaceMatrix(fe1, *fe2_ptr, *Tr, elmat);

          shared_fes.GetElementDofs(e1, shared_dofs);
          MFEM_VERIFY(shared_dofs.Size() == shared_dof,
                      "Shared exact-face apply local dof mismatch.");
          for (int d = 0; d < shared_dof; ++d)
          {
            int vdof = shared_fes.DofToVDof(shared_dofs[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            xloc(d) = sigma_data[vdof];
          }

          MFEM_VERIFY(shared_fes.GetFaceNbrElementVDofs(nbr_el,
                                                        nbr_dofs_ref) ==
                        nullptr,
                      "Shared exact-face apply test does not support "
                      "neighbor dof transformations.");
          MFEM_VERIFY(nbr_dofs_ref.Size() == shared_dof,
                      "Shared exact-face apply neighbor dof mismatch.");
          for (int d = 0; d < shared_dof; ++d)
          {
            const int vdof = nbr_dofs_ref[d];
            xloc(shared_dof + d) =
              (vdof >= 0) ? sigma_nbr_data[vdof]
                          : -sigma_nbr_data[-1 - vdof];
          }

          elmat.Mult(xloc, yloc);

          for (int d = 0; d < shared_dof; ++d)
          {
            int vdof = shared_fes.DofToVDof(shared_dofs[d], 0);
            vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
            assembled_data[vdof] += yloc(d);
            assembled_shared_diag_data[vdof] += elmat(d, d);
          }
        }

        mfem::Vector shared_delta(shared_apply_y);
        shared_delta -= assembled_shared_y;
        const double shared_ref_norm =
          std::max(GlobalL2Norm(assembled_shared_y), 1.0e-30);
        const double shared_rel =
          GlobalL2Norm(shared_delta) / shared_ref_norm;
        const double* shared_delta_data = shared_delta.HostRead();
        double local_shared_max_abs = 0.0;
        for (int i = 0; i < shared_delta.Size(); ++i)
        {
          local_shared_max_abs =
            std::max(local_shared_max_abs, std::abs(shared_delta_data[i]));
        }
        double global_shared_max_abs = 0.0;
        MPI_Allreduce(&local_shared_max_abs,
                      &global_shared_max_abs,
                      1,
                      MPI_DOUBLE,
                      MPI_MAX,
                      MPI_COMM_WORLD);
        ctx.LogMetric("igr_l2_exact_face_shared_apply_local_qf rel=" +
                      FormatSci(shared_rel) +
                      " max_abs=" + FormatSci(global_shared_max_abs) +
                      " ref_norm=" + FormatSci(shared_ref_norm) +
                      " nf=" + std::to_string(shared_apply_info.nf) +
                      " nq=" + std::to_string(shared_apply_info.nq));
        ctx.Check(shared_rel < 5.0e-13,
                  "IGR exact-face shared local apply matches assembled MFEM");
        ctx.Check(global_shared_max_abs < 5.0e-13,
                  "IGR exact-face shared local apply max error is tiny");

        mfem::Vector shared_diag_delta(shared_diag);
        shared_diag_delta -= assembled_shared_diag;
        const double shared_diag_ref_norm =
          std::max(GlobalL2Norm(assembled_shared_diag), 1.0e-30);
        const double shared_diag_rel =
          GlobalL2Norm(shared_diag_delta) / shared_diag_ref_norm;
        const double* shared_diag_delta_data = shared_diag_delta.HostRead();
        double local_shared_diag_max_abs = 0.0;
        for (int i = 0; i < shared_diag_delta.Size(); ++i)
        {
          local_shared_diag_max_abs =
            std::max(local_shared_diag_max_abs,
                     std::abs(shared_diag_delta_data[i]));
        }
        double global_shared_diag_max_abs = 0.0;
        MPI_Allreduce(&local_shared_diag_max_abs,
                      &global_shared_diag_max_abs,
                      1,
                      MPI_DOUBLE,
                      MPI_MAX,
                      MPI_COMM_WORLD);
        ctx.LogMetric("igr_l2_exact_face_shared_diag_qf rel=" +
                      FormatSci(shared_diag_rel) +
                      " max_abs=" + FormatSci(global_shared_diag_max_abs) +
                      " ref_norm=" + FormatSci(shared_diag_ref_norm));
        ctx.Check(shared_diag_rel < 5.0e-13,
                  "IGR exact-face shared local diagonal matches assembled MFEM");
        ctx.Check(global_shared_diag_max_abs < 5.0e-13,
                  "IGR exact-face shared local diagonal max error is tiny");
      }
    }

    int local_exact_operator_bad = 0;
    int local_exact_operator_shared_nf = 0;
    int local_exact_operator_local_nf = 0;
    int local_exact_operator_converged = 0;
    int local_harmonic_operator_bad = 0;
    int local_harmonic_operator_converged = 0;
    double harmonic_symmetry_defect = 0.0;
    if (local_pmesh_shared > 0)
    {
      constexpr int num_eq_shared = dim_shared + 2;
      mfem::ParFiniteElementSpace shared_state_fes(&pmesh_shared,
                                                   &shared_fec,
                                                   num_eq_shared,
                                                   mfem::Ordering::byNODES);
      mfem::ParGridFunction shared_state(&shared_state_fes);
      const int rank_shared = mfem::Mpi::WorldRank();
      mfem::VectorFunctionCoefficient shared_state_coeff(
        num_eq_shared,
        [rank_shared](const mfem::Vector& x, mfem::Vector& cons)
        {
          const double rho =
            1.0 + 0.04 * std::sin(2.0 * x(0)) +
            0.03 * std::cos(3.0 * x(1)) +
            0.01 * static_cast<double>(rank_shared);
          const double u = 0.25 + 0.02 * x(1);
          const double v = -0.15 + 0.03 * x(0);
          const double p = 1.0 + 0.01 * x(0) * x(1);
          cons = EulerConsFromPrim2D(rho, u, v, p, /*gamma=*/1.4);
        });
      shared_state.ProjectCoefficient(shared_state_coeff);
      shared_state.ExchangeFaceNbrData();

      ScopedEnvVar force_pa("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
      ScopedEnvVar force_qf("DG_IGR_IGR_DEVICE_QF", "1");
      ScopedEnvVar force_exact("DG_IGR_IGR_EXACT_DG_FACE_OPERATOR", "1");
      ScopedEnvVar enable_extended_certificate(
        "DG_IGR_IGR_EXTENDED_CERTIFICATE", "1");
      ScopedEnvVar force_certificate_start(
        "DG_IGR_IGR_EXTENDED_CERTIFICATE_START", "1");
      ScopedEnvVar force_certificate_factor(
        "DG_IGR_IGR_EXTENDED_CERTIFICATE_FACTOR", "100");
      igrlab::IGRSolver shared_exact_solver(shared_fes,
                                            shared_state_fes,
                                            dim_shared,
                                            num_eq_shared,
                                            /*C_alpha=*/0.19);
      shared_exact_solver.SetAlphaLengthScalePolicy(
        igrlab::IGRAlphaLengthScalePolicy::Local);
      shared_exact_solver.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
      shared_exact_solver.SetPreconditioner(igrlab::IGRPrecondType::None);
      shared_exact_solver.SetLinearSolveParams(/*rtol=*/1.0e-9,
                                               /*atol=*/0.0,
                                               /*max_it=*/400,
                                               /*print=*/0);
      shared_exact_solver.UpdateFromState(shared_state, /*warm_start=*/false);
      local_exact_operator_converged =
        shared_exact_solver.GetLastLinearConverged() ? 1 : 0;
      local_exact_operator_shared_nf =
        shared_exact_solver.GetLastExactFaceSharedFaceCount();
      local_exact_operator_local_nf =
        shared_exact_solver.GetLastExactFaceLocalFaceCount();
      if (!shared_exact_solver.GetLastLinearConverged() ||
          !shared_exact_solver.GetLastExactFaceEffective() ||
          local_exact_operator_shared_nf != local_pmesh_shared ||
          local_exact_operator_local_nf <= 0)
      {
        local_exact_operator_bad = 1;
      }
      if (!std::isfinite(
            shared_exact_solver.GetLastLinearTrueResidualRelRHS()))
      {
        local_exact_operator_bad = 1;
      }

      {
        ScopedEnvVar reject_certificate_factor(
          "DG_IGR_IGR_EXTENDED_CERTIFICATE_FACTOR", "1e-12");
        igrlab::IGRSolver certificate_rejection_solver(shared_fes,
                                                       shared_state_fes,
                                                       dim_shared,
                                                       num_eq_shared,
                                                       /*alpha=*/0.19);
        certificate_rejection_solver.SetLinearSolver(
          igrlab::IGRLinearSolverType::CG);
        certificate_rejection_solver.SetPreconditioner(
          igrlab::IGRPrecondType::None);
        certificate_rejection_solver.SetNonconvergencePolicy(
          igrlab::IGRNonconvergencePolicy::Accept);
        certificate_rejection_solver.SetLinearSolveParams(/*rtol=*/1.0e-9,
                                                          /*atol=*/0.0,
                                                          /*max_it=*/400,
                                                          /*print=*/0);
        certificate_rejection_solver.UpdateFromState(
          shared_state, /*warm_start=*/false);
        if (certificate_rejection_solver.GetLastLinearConverged())
        {
          local_exact_operator_bad = 1;
        }
      }

      {
        ScopedEnvVar force_harmonic(
          "DG_IGR_IGR_IPDG_FACE_SCALING", "harmonic");
        igrlab::IGRSolver harmonic_solver(shared_fes,
                                          shared_state_fes,
                                          dim_shared,
                                          num_eq_shared,
                                          /*alpha=*/0.19);
        harmonic_solver.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
        harmonic_solver.SetPreconditioner(igrlab::IGRPrecondType::None);
        harmonic_solver.SetLinearSolveParams(/*rtol=*/1.0e-9,
                                             /*atol=*/0.0,
                                             /*max_it=*/400,
                                             /*print=*/0);
        harmonic_solver.UpdateFromState(shared_state, /*warm_start=*/false);
        local_harmonic_operator_converged =
          harmonic_solver.GetLastLinearConverged() ? 1 : 0;

        const mfem::Operator* harmonic_operator =
          harmonic_solver.GetLinearOperator();
        if (harmonic_operator == nullptr ||
            harmonic_solver.GetLastExactFaceSharedFaceCount() !=
              local_pmesh_shared)
        {
          local_harmonic_operator_bad = 1;
        }
        else
        {
          mfem::Vector x(harmonic_operator->Width());
          mfem::Vector y(harmonic_operator->Width());
          mfem::Vector ax(harmonic_operator->Height());
          mfem::Vector ay(harmonic_operator->Height());
          for (int i = 0; i < x.Size(); ++i)
          {
            const double index = static_cast<double>(
              i + 1 + 104729 * mfem::Mpi::WorldRank());
            x(i) = std::sin(0.017 * index) + 0.31 * std::cos(0.023 * index);
            y(i) = std::cos(0.019 * index) - 0.27 * std::sin(0.029 * index);
          }
          harmonic_operator->Mult(x, ax);
          harmonic_operator->Mult(y, ay);
          double products[2] = {x * ay, y * ax};
          MPI_Allreduce(MPI_IN_PLACE,
                        products,
                        2,
                        mfem::MPITypeMap<double>::mpi_type,
                        MPI_SUM,
                        MPI_COMM_WORLD);
          harmonic_symmetry_defect =
            std::abs(products[0] - products[1]) /
            std::max({std::abs(products[0]), std::abs(products[1]), 1.0});
          if (harmonic_symmetry_defect > 1.0e-11)
          {
            local_harmonic_operator_bad = 1;
          }
        }
        if (!harmonic_solver.GetLastLinearConverged())
        {
          local_harmonic_operator_bad = 1;
        }
      }
    }

    const int local_bad_support =
      (local_shared_in_fqs > 0 &&
       (!side_supported || !trace_supported)) ? 1 : 0;
    int global_bad_support = 0;
    int global_shared_apply_bad = 0;
    int global_shared_repeat_bad = 0;
    int global_shared_apply_nf = 0;
    int global_exact_operator_bad = 0;
    int global_exact_operator_shared_nf = 0;
    int global_exact_operator_local_nf = 0;
    int global_exact_operator_converged = 0;
    int global_harmonic_operator_bad = 0;
    int global_harmonic_operator_converged = 0;
    int global_shared_alpha_bad = 0;
    double global_shared_alpha_error = 0.0;
    double global_shared_alpha_jump = 0.0;
    double global_shared_alpha_vs_max = 0.0;
    MPI_Allreduce(&local_bad_support,
                  &global_bad_support,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_shared_apply_bad,
                  &global_shared_apply_bad,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_shared_repeat_bad,
                  &global_shared_repeat_bad,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_shared_apply_nf,
                  &global_shared_apply_nf,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_exact_operator_bad,
                  &global_exact_operator_bad,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_exact_operator_shared_nf,
                  &global_exact_operator_shared_nf,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_exact_operator_local_nf,
                  &global_exact_operator_local_nf,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_exact_operator_converged,
                  &global_exact_operator_converged,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_harmonic_operator_bad,
                  &global_harmonic_operator_bad,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_harmonic_operator_converged,
                  &global_harmonic_operator_converged,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_shared_alpha_bad,
                  &global_shared_alpha_bad,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_shared_alpha_error,
                  &global_shared_alpha_error,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_shared_alpha_jump,
                  &global_shared_alpha_jump,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_shared_alpha_vs_max,
                  &global_shared_alpha_vs_max,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);

    ctx.LogMetric("igr_l2_exact_face_shared_guard local_shared_fqs=" +
                  std::to_string(local_shared_in_fqs) +
                  " global_shared_fqs=" +
                  std::to_string(global_shared_in_fqs) +
                  " global_pmesh_shared=" +
                  std::to_string(global_pmesh_shared) +
                  " side_supported=" +
                  std::to_string(side_supported ? 1 : 0) +
                  " trace_supported=" +
                  std::to_string(trace_supported ? 1 : 0) +
                  " shared_apply_supported=" +
                  std::to_string(shared_apply_supported ? 1 : 0) +
                  " global_shared_apply_nf=" +
                  std::to_string(global_shared_apply_nf));
    ctx.LogMetric("igr_l2_exact_face_mpi_operator_shared local_nf=" +
                  std::to_string(local_exact_operator_local_nf) +
                  " global_local_nf=" +
                  std::to_string(global_exact_operator_local_nf) +
                  " shared_nf=" +
                  std::to_string(local_exact_operator_shared_nf) +
                  " global_shared_nf=" +
                  std::to_string(global_exact_operator_shared_nf) +
                  " converged_ranks=" +
                  std::to_string(global_exact_operator_converged));
    ctx.LogMetric("igr_l2_exact_face_harmonic_mpi symmetry=" +
                  FormatSci(harmonic_symmetry_defect) +
                  " converged_ranks=" +
                  std::to_string(global_harmonic_operator_converged));
    ctx.LogMetric("igr_local_alpha_shared_face_qf err=" +
                  FormatSci(global_shared_alpha_error) +
                  " side_jump=" + FormatSci(global_shared_alpha_jump) +
                  " local_vs_max=" +
                  FormatSci(global_shared_alpha_vs_max));
    ctx.AddParity(static_cast<double>(global_shared_in_fqs +
                                      global_pmesh_shared +
                                      global_bad_support +
                                      global_shared_apply_bad +
                                      global_shared_repeat_bad +
                                      global_shared_apply_nf +
                                      global_exact_operator_bad +
                                      global_exact_operator_shared_nf +
                                      global_harmonic_operator_bad));
    ctx.Check(global_pmesh_shared > 0,
              "IGR exact-face MPI guard test creates shared ParMesh faces");
    ctx.Check(global_shared_in_fqs > 0,
              "IGR exact-face MPI guard sees shared faces in FaceQuadratureSpace");
    ctx.Check(global_bad_support == 0,
              "IGR exact-face local helpers support parallel face spaces");
    ctx.Check(global_shared_apply_bad == 0,
              "IGR exact-face shared-face metadata builder is valid");
    ctx.Check(global_shared_repeat_bad == 0,
              "IGR exact-face shared row gather repeats bit-for-bit");
    ctx.Check(global_shared_apply_nf == global_pmesh_shared,
              "IGR exact-face shared-face metadata covers ParMesh shared faces");
    ctx.Check(global_shared_alpha_bad == 0 &&
                global_shared_alpha_error < 5.0e-13,
              "IGR shared-face qdata preserves exact local and neighbor "
              "C_alpha*h_e^2 values");
    ctx.Check(global_shared_alpha_jump > 1.0e-3 &&
                global_shared_alpha_vs_max > 1.0e-3,
              "IGR shared-face local alpha crosses unequal elements and is "
              "not constant alpha_max");
    ctx.Check(global_exact_operator_bad == 0,
              "IGR exact-face MPI operator path converges with shared faces");
    ctx.Check(global_exact_operator_shared_nf == global_pmesh_shared,
              "IGR exact-face MPI operator constructs shared face data");
    ctx.Check(global_harmonic_operator_bad == 0,
              "IGR harmonic exact-face MPI operator is symmetric and converges");
  }

  // --- Constant-density locality: alpha=h^2 keeps response width in cells ---
  {
    struct LocalityResult
    {
      double radius_over_h = 0.0;
      double sigma_max = 0.0;
    };
    auto MeasureLocality = [&](int elements)
    {
      constexpr int locality_dim = 1;
      constexpr int locality_num_eq = locality_dim + 2;
      const double h = 1.0 / static_cast<double>(elements);
      mfem::Mesh locality_mesh = MakePeriodicMesh1D(elements);
      mfem::ParMesh locality_pmesh(MPI_COMM_SELF, locality_mesh);
      mfem::DG_FECollection locality_state_fec(
        2, locality_dim, mfem::BasisType::GaussLegendre);
      mfem::ParFiniteElementSpace locality_state_fes(
        &locality_pmesh,
        &locality_state_fec,
        locality_num_eq,
        mfem::Ordering::byNODES);
      mfem::ParGridFunction locality_state(&locality_state_fes);
      mfem::VectorFunctionCoefficient locality_state_coefficient(
        locality_num_eq,
        [h](const mfem::Vector& x, mfem::Vector& conservative)
        {
          const double distance = std::abs(x(0) - 0.5);
          const double width = 2.0 * h;
          const double velocity =
            std::exp(-(distance * distance) / (width * width));
          mfem::Vector primitive(3);
          primitive(0) = 1.0;
          primitive(1) = velocity;
          primitive(2) = 1.0;
          mfem::EulerPrimToConservative(
            primitive, /*dim=*/1, /*gamma=*/1.4, conservative);
        });
      locality_state.ProjectCoefficient(locality_state_coefficient);

      mfem::DG_FECollection locality_sigma_fec(
        1, locality_dim, mfem::BasisType::GaussLegendre);
      mfem::ParFiniteElementSpace locality_sigma_fes(
        &locality_pmesh,
        &locality_sigma_fec,
        1,
        mfem::Ordering::byNODES);
      igrlab::IGRSolver locality_solver(locality_sigma_fes,
                                        locality_state_fes,
                                        locality_dim,
                                        locality_num_eq,
                                        h * h);
      locality_solver.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
      locality_solver.SetSolveMode(igrlab::IGRSolveMode::Krylov);
      locality_solver.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
      locality_solver.SetPreconditioner(igrlab::IGRPrecondType::None);
      locality_solver.SetLinearSolveParams(1.0e-10, 0.0, 1000, 0);
      locality_solver.UpdateFromState(locality_state, /*warm_start=*/false);
      ctx.Check(locality_solver.GetLastLinearConverged(),
                "IGR constant-density locality solve converges at n=" +
                  std::to_string(elements));

      const mfem::ParGridFunction& locality_sigma =
        locality_solver.GetSigma();
      const mfem::IntegrationPoint& center =
        mfem::Geometries.GetCenter(mfem::Geometry::SEGMENT);
      double weighted_distance2 = 0.0;
      double total_weight = 0.0;
      double sigma_max = 0.0;
      for (int e = 0; e < locality_sigma_fes.GetNE(); ++e)
      {
        mfem::ElementTransformation* transformation =
          locality_sigma_fes.GetElementTransformation(e);
        MFEM_VERIFY(transformation,
                    "IGR locality test requires element transformation.");
        transformation->SetIntPoint(&center);
        mfem::Vector physical(1);
        transformation->Transform(center, physical);
        const double sigma_value =
          locality_sigma.GetValue(*transformation, center);
        const double weight = std::abs(sigma_value) * h;
        const double distance = std::abs(physical(0) - 0.5);
        weighted_distance2 += weight * distance * distance;
        total_weight += weight;
        sigma_max = std::max(sigma_max, std::abs(sigma_value));
      }
      MFEM_VERIFY(total_weight > 0.0,
                  "IGR locality test requires nonzero sigma response.");
      return LocalityResult{
        std::sqrt(weighted_distance2 / total_weight) / h,
        sigma_max};
    };

    const LocalityResult locality_32 = MeasureLocality(32);
    const LocalityResult locality_64 = MeasureLocality(64);
    const double locality_radius_ratio =
      locality_64.radius_over_h / locality_32.radius_over_h;
    const double locality_amplitude_ratio =
      locality_64.sigma_max / locality_32.sigma_max;
    ctx.LogMetric("igr_locality_alpha_h2 n32_radius_over_h=" +
                  FormatSci(locality_32.radius_over_h) +
                  " n64_radius_over_h=" +
                  FormatSci(locality_64.radius_over_h) +
                  " radius_ratio=" + FormatSci(locality_radius_ratio) +
                  " amplitude_ratio=" +
                  FormatSci(locality_amplitude_ratio));
    ctx.Check(std::isfinite(locality_radius_ratio) &&
                locality_radius_ratio > 0.65 &&
                locality_radius_ratio < 1.55,
              "IGR alpha=h^2 response radius remains bounded in cell counts");
    ctx.Check(std::isfinite(locality_amplitude_ratio) &&
                locality_amplitude_ratio > 0.35 &&
                locality_amplitude_ratio < 2.5,
              "IGR alpha=h^2 response amplitude remains mesh-bounded");
  }

  // --- Quasi-1D MPI robustness: sigma should vary primarily in x, not y ---
  {
    constexpr int dim_q = 2;
    constexpr int num_eq_q = dim_q + 2;
    constexpr int nx_q = 24;
    constexpr int ny_q = 6;
    constexpr double gamma_q = 1.4;

    mfem::Mesh mesh_q = MakePeriodicMesh2D(nx_q, ny_q, /*sx=*/1.0, /*sy=*/0.25);
    mfem::ParMesh pmesh_q(MPI_COMM_WORLD, mesh_q);

    mfem::DG_FECollection state_fec_q(/*order=*/2,
                                      dim_q,
                                      mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace state_fes_q(&pmesh_q,
                                            &state_fec_q,
                                            num_eq_q,
                                            mfem::Ordering::byNODES);
    mfem::ParGridFunction U_q(&state_fes_q);
    mfem::VectorFunctionCoefficient U_coeff_q(
      num_eq_q,
      [&](const mfem::Vector& x, mfem::Vector& cons)
      {
        // Quasi-1D stationary-shock-like profile in x only.
        const double xc = 0.5;
        const double w = 0.04;
        const double s = 0.5 * (1.0 - std::tanh((x(0) - xc) / w));
        const double rho = 1.0 + 1.2 * s;
        const double u = -2.2 + 1.6 * s;
        const double v = 0.0;
        const double p = 0.8 + 1.0 * s;
        cons = EulerConsFromPrim2D(rho, u, v, p, gamma_q);
      });
    U_q.ProjectCoefficient(U_coeff_q);
    U_q.ExchangeFaceNbrData();

    mfem::DG_FECollection sigma_fec_q(/*order=*/1,
                                      dim_q,
                                      mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace sigma_fes_q(&pmesh_q,
                                            &sigma_fec_q,
                                            /*vdim=*/1,
                                            mfem::Ordering::byNODES);

    igrlab::IGRSolver solver_q(sigma_fes_q,
                               state_fes_q,
                               dim_q,
                               num_eq_q,
                               /*alpha=*/0.5);
    solver_q.SetSolveMode(igrlab::IGRSolveMode::Krylov);
    solver_q.SetLinearSolver(igrlab::IGRLinearSolverType::Auto);
    solver_q.SetPreconditioner(igrlab::IGRPrecondType::Auto);
    solver_q.SetLinearSolveParams(/*rtol=*/1.0e-8,
                                  /*atol=*/0.0,
                                  /*max_it=*/500,
                                  /*print=*/0);
    solver_q.UpdateFromState(U_q, /*warm_start=*/false);

    ctx.Check(solver_q.GetLastLinearConverged(),
              "IGR quasi-1D L2 solve converges");

    const mfem::ParGridFunction& sigma_q = solver_q.GetSigma();
    const mfem::ParFiniteElementSpace* sfes_q = sigma_q.ParFESpace();
    MFEM_VERIFY(sfes_q != nullptr,
                "IGR quasi-1D test requires sigma FE space.");

    std::vector<double> local_col_min(nx_q,
                                      std::numeric_limits<double>::infinity());
    std::vector<double> local_col_max(nx_q,
                                      -std::numeric_limits<double>::infinity());
    std::vector<double> local_col_sum(nx_q, 0.0);
    std::vector<double> local_col_count(nx_q, 0.0);

    mfem::IntegrationPoint ip_center;
    ip_center.Set2(0.0, 0.0);
    for (int e = 0; e < sfes_q->GetNE(); ++e)
    {
      const long long gid = pmesh_q.GetGlobalElementNum(e);
      const int ix = static_cast<int>(gid % nx_q);
      mfem::ElementTransformation* Te = sfes_q->GetElementTransformation(e);
      MFEM_VERIFY(Te != nullptr,
                  "IGR quasi-1D test failed to get element transformation.");
      Te->SetIntPoint(&ip_center);
      const double sigma_val = sigma_q.GetValue(*Te, ip_center);

      local_col_min[ix] = std::min(local_col_min[ix], sigma_val);
      local_col_max[ix] = std::max(local_col_max[ix], sigma_val);
      local_col_sum[ix] += sigma_val;
      local_col_count[ix] += 1.0;
    }

    std::vector<double> global_col_min(nx_q, 0.0);
    std::vector<double> global_col_max(nx_q, 0.0);
    std::vector<double> global_col_sum(nx_q, 0.0);
    std::vector<double> global_col_count(nx_q, 0.0);

    MPI_Allreduce(local_col_min.data(),
                  global_col_min.data(),
                  nx_q,
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_col_max.data(),
                  global_col_max.data(),
                  nx_q,
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_col_sum.data(),
                  global_col_sum.data(),
                  nx_q,
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_col_count.data(),
                  global_col_count.data(),
                  nx_q,
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    double max_y_spread = 0.0;
    double x_min_avg = std::numeric_limits<double>::infinity();
    double x_max_avg = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < nx_q; ++i)
    {
      if (global_col_count[i] <= 0.0)
      {
        continue;
      }
      max_y_spread =
        std::max(max_y_spread, global_col_max[i] - global_col_min[i]);
      const double avg = global_col_sum[i] / global_col_count[i];
      x_min_avg = std::min(x_min_avg, avg);
      x_max_avg = std::max(x_max_avg, avg);
    }
    const double x_span = x_max_avg - x_min_avg;
    const double y_to_x_ratio = max_y_spread / std::max(x_span, 1.0e-14);
    ctx.LogMetric("igr_quasi1d_sigma_spread max_y_spread=" +
                  FormatSci(max_y_spread) + " x_span=" + FormatSci(x_span) +
                  " y_to_x_ratio=" + FormatSci(y_to_x_ratio));
    ctx.AddParity(max_y_spread + x_span + y_to_x_ratio);

    ctx.Check(x_span > 1.0e-6,
              "IGR quasi-1D sigma has non-trivial x variation");
    ctx.Check(std::isfinite(y_to_x_ratio) && y_to_x_ratio < 20.0,
              "IGR quasi-1D sigma y-variation remains bounded relative to x");
  }

  return ctx.failures;
}

} // namespace igrlab_tests

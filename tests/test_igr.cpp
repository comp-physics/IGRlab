#include "physics/euler/conversions.hpp"
#include "physics/igr/IGR.hpp"
#include "physics/igr/igr_coupling.hpp"
#include "physics/igr/igr_flux.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/riemann/side_scope_utils.hpp"
#include "../src/numerics/dg/DGFaceQuadratureKernels.hpp"

#include <algorithm>
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
  dg_igr::dg_face::InteriorFaceSideDensityProjectionScratch coeff_scratch;
  const auto coeff_info =
    dg_igr::dg_face::ProjectInteriorFaceDensityCoefficient2D(state_fes,
                                                             fqs,
                                                             state,
                                                             alpha,
                                                             rho_floor,
                                                             side_q,
                                                             &coeff_scratch);

  mfem::Vector exact_face_y(sigma.Size());
  exact_face_y = 0.0;
  dg_igr::dg_face::InteriorFaceDGDiffusionApplyScratch apply_scratch;
  const auto apply_info =
    dg_igr::dg_face::AddInteriorFaceDGDiffusionApply2D(sigma_fes,
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
  const double rho_floor = mfem::dg_igr::FluxDefaults::kRhoVelFloor;
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
    const char* prev = std::getenv(name_.c_str());
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

  mfem::EulerFlux base_flux(dim, gamma);
  mfem::FluxWithIGR igr_flux(base_flux);

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
    base_flux.ComputeFluxDotN(state, normal, Tr_dummy, base_fn);
    igr_flux.ComputeFluxDotN(state, normal, Tr_dummy, igr_fn);
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

  mfem::FaceElementTransformations& Tr_face = GetFaceTr(pmesh, /*face=*/0);

  mfem::Vector base_fn(dim + 2);
  mfem::Vector igr_fn(dim + 2);
  base_flux.ComputeFluxDotN(state, normal, Tr_face, base_fn);
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

      const double max_speed = std::max(speed1, speed2);
      const double scaled_max_speed = max_speed * normal_if.Norml2();

      mfem::Vector expected(dim + 2);
      for (int i = 0; i < expected.Size(); ++i)
      {
        expected(i) =
          0.5 * (scaled_max_speed * (state_left(i) - state_right(i)) +
                 fluxN1(i) + fluxN2(i));
      }

      mfem::Vector actual(dim + 2);
      num_flux.Eval(state_left, state_right, normal_if, *Tr, actual);

      for (int i = 0; i < actual.Size(); ++i)
      {
        local_max_err =
          std::max(local_max_err, std::abs(actual(i) - expected(i)));
      }
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

    dg_igr::IGRSolver solver_a(sigma_fes,
                               state_fes,
                               igr_dim,
                               igr_num_eq,
                               /*alpha=*/0.05);
    dg_igr::IGRSolver solver_b(sigma_fes,
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
    dg_igr::IGRSolver h1_solver(sigma_h1_fes,
                                state_fes_ws,
                                dim_ws,
                                num_eq_ws,
                                /*alpha=*/0.5);
    h1_solver.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
    h1_solver.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
    h1_solver.SetLinearSolver(dg_igr::IGRLinearSolverType::Auto);
    h1_solver.SetPreconditioner(dg_igr::IGRPrecondType::Auto);
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
      dg_igr::IGRSolver pa_boomer_solver(sigma_h1_fes,
                                         state_fes_ws,
                                         dim_ws,
                                         num_eq_ws,
                                         /*alpha=*/0.5);
      pa_boomer_solver.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
      pa_boomer_solver.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
      pa_boomer_solver.SetLinearSolver(dg_igr::IGRLinearSolverType::CG);
      pa_boomer_solver.SetPreconditioner(dg_igr::IGRPrecondType::BoomerAMG);
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

    // L2 auto policy with SIPG should be SPD-by-construction -> CG + BoomerAMG.
    mfem::DG_FECollection sigma_l2_fec(/*order=*/1,
                                       dim_ws,
                                       mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace sigma_l2_fes(&pmesh_ws,
                                             &sigma_l2_fec,
                                             /*vdim=*/1,
                                             mfem::Ordering::byNODES);
    dg_igr::IGRSolver l2_solver(sigma_l2_fes,
                                state_fes_ws,
                                dim_ws,
                                num_eq_ws,
                                /*alpha=*/0.5);
    l2_solver.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
    l2_solver.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
    l2_solver.SetLinearSolver(dg_igr::IGRLinearSolverType::Auto);
    l2_solver.SetPreconditioner(dg_igr::IGRPrecondType::Auto);
    l2_solver.SetLinearSolveParams(/*rtol=*/1.0e-10,
                                   /*atol=*/0.0,
                                   /*max_it=*/250,
                                   /*print=*/0);
    project_state(/*phase=*/0.0);
    l2_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(l2_solver.GetLastLinearConverged(),
              "IGR L2 auto policy converges");
    ctx.Check(l2_solver.GetLastLinearSolverName() == "cg",
              "IGR L2 SIPG auto policy selects CG");
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
          dg_igr::dg_face::SupportsInteriorFaceSideDensityProjection2D(
            state_fes_qf,
            side_fqs,
            &side_reason);
        ctx.Check(side_supported,
                  "IGR side-density projection supports serial 2D L2 faces" +
                    (side_reason.empty() ? std::string()
                                         : (" (" + side_reason + ")")));

        if (side_supported)
        {
          dg_igr::dg_face::InteriorFaceSideDensityProjectionScratch scratch;
          mfem::Vector side_q;
          const double alpha_side = 0.37;
          const double rho_floor_side = 0.2;
          const auto side_info =
            dg_igr::dg_face::ProjectInteriorFaceDensityCoefficient2D(
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
          dg_igr::dg_face::SupportsInteriorFaceScalarTraceProjection2D(
            sigma_fes_qf,
            trace_fqs,
            &trace_reason);
        ctx.Check(trace_supported,
                  "IGR scalar trace projection supports serial 2D L2 faces" +
                    (trace_reason.empty() ? std::string()
                                          : (" (" + trace_reason + ")")));

        if (trace_supported)
        {
          dg_igr::dg_face::InteriorFaceScalarTraceProjectionScratch scratch;
          mfem::Vector trace_value_q;
          mfem::Vector trace_normal_q;
          const auto trace_info =
            dg_igr::dg_face::ProjectInteriorFaceScalarTrace2D(
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
                  dg_igr::dg_face::ReferenceNormalComponent2D(local_face_id);
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
        dg_igr::dg_face::InteriorFaceSideDensityProjectionScratch coeff_scratch;
        const auto coeff_info =
          dg_igr::dg_face::ProjectInteriorFaceDensityCoefficient2D(
            state_fes_qf,
            apply_fqs,
            U_face_apply,
            alpha_apply,
            rho_floor_apply,
            side_q_apply,
            &coeff_scratch);

        const double ip_sigma_apply =
          static_cast<double>(dg_igr::IGRIPKind::SIPG);
        const double kappa_apply = 2.75;
        mfem::Vector exact_face_y(sigma_face_apply.Size());
        exact_face_y = 0.0;
        dg_igr::dg_face::InteriorFaceDGDiffusionApplyScratch apply_scratch;
        const auto apply_info =
          dg_igr::dg_face::AddInteriorFaceDGDiffusionApply2D(
            sigma_fes_qf,
            apply_fqs,
            sigma_face_apply,
            side_q_apply,
            ip_sigma_apply,
            kappa_apply,
            exact_face_y,
            &apply_scratch);

        ctx.Check(coeff_info.qf_size == 2 * apply_info.nq * apply_info.nf,
                  "IGR exact face apply receives matching side-coefficient qdata");
        ctx.Check(apply_info.nf > 0,
                  "IGR exact face apply sees interior faces");

        mfem::Vector exact_face_diag(sigma_face_apply.Size());
        exact_face_diag = 0.0;
        dg_igr::dg_face::AddInteriorFaceDGDiffusionDiagonal2D(
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
                                                   dg_igr::IGRIPKind::SIPG),
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
        [&](dg_igr::IGRSolver& solver)
      {
        solver.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
        solver.SetIPKind(dg_igr::IGRIPKind::SIPG);
        solver.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
        solver.SetLinearSolver(dg_igr::IGRLinearSolverType::CG);
        solver.SetPreconditioner(dg_igr::IGRPrecondType::None);
        solver.SetLinearSolveParams(/*rtol=*/1.0e-11,
                                    /*atol=*/0.0,
                                    /*max_it=*/1000,
                                    /*print=*/0);
      };

      mfem::H1_FECollection sigma_h1_fec_qf(/*order=*/1, dim_qf);
      mfem::ParFiniteElementSpace sigma_h1_fes_qf(&pmesh_qf,
                                                  &sigma_h1_fec_qf,
                                                  /*vdim=*/1);
      dg_igr::IGRSolver assembled_h1_qf_ref(sigma_h1_fes_qf,
                                            state_fes_qf,
                                            dim_qf,
                                            num_eq_qf,
                                            /*alpha=*/0.3);
      configure_qf_solver(assembled_h1_qf_ref);
      assembled_h1_qf_ref.UpdateFromState(U_qf, /*warm_start=*/false);
	      {
	        ScopedEnvVar force_pa("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
	        ScopedEnvVar force_qf("DG_IGR_IGR_DEVICE_QF", "1");
        dg_igr::IGRSolver pa_h1_qf_ref(sigma_h1_fes_qf,
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

      dg_igr::IGRSolver assembled_qf_ref(sigma_fes_qf,
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
	                                              mfem::dg_igr::IGRDefaults::
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
	                       mfem::dg_igr::IGRDefaults::kRhoFloor);
	            invrho_vals(q, e) = 1.0 / rho;
	            ck_vals(q, e) = 0.3 / rho;
	            src_eval.StateScratch() = state_at_q;
	            src_eval.GradientScratch() = grad_at_q;
	            src_vals(q, e) = src_eval.EvalFromStateAndGradient();
	          }
	        }

	        InverseRhoCoefficient invrho_cb(
	          U_qf,
	          mfem::dg_igr::IGRDefaults::kRhoFloor);
	        AlphaOverRhoCoefficient ck_cb(
	          U_qf,
	          /*alpha=*/0.3,
	          mfem::dg_igr::IGRDefaults::kRhoFloor);
	        BrokenDGIGRSourceCoefficient src_cb(
	          U_qf,
	          dim_qf,
	          num_eq_qf,
	          /*alpha=*/0.3,
	          mfem::dg_igr::IGRDefaults::kRhoFloor);
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
        dg_igr::IGRSolver cheb_qf_ref(sigma_fes_qf,
                                      state_fes_qf,
                                      dim_qf,
                                      num_eq_qf,
                                      /*alpha=*/0.3);
        cheb_qf_ref.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
        cheb_qf_ref.SetIPKind(dg_igr::IGRIPKind::SIPG);
        cheb_qf_ref.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
        cheb_qf_ref.SetLinearSolver(dg_igr::IGRLinearSolverType::CG);
        cheb_qf_ref.SetPreconditioner(
          dg_igr::IGRPrecondType::OperatorChebyshev);
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
        dg_igr::IGRSolver pa_qf_ref(sigma_fes_qf,
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
          std::getenv("DG_IGR_REQUIRE_L2_PA_QF_EQUIV") != nullptr;
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
	        ScopedEnvVar force_pa("DG_IGR_IGR_PARTIAL_ASSEMBLY", "1");
	        ScopedEnvVar force_qf("DG_IGR_IGR_DEVICE_QF", "1");
	        ScopedEnvVar force_exact("DG_IGR_IGR_EXACT_DG_FACE_OPERATOR", "1");
	        dg_igr::IGRSolver exact_face_ref(sigma_fes_qf,
	                                         state_fes_qf,
	                                         dim_qf,
                                         num_eq_qf,
                                         /*alpha=*/0.3);
        configure_qf_solver(exact_face_ref);
        exact_face_ref.UpdateFromState(U_qf, /*warm_start=*/false);
        ctx.Check(exact_face_ref.GetLastLinearConverged(),
                  "IGR CPU L2 exact-face PA/QF reference solve converges");
        ctx.Check(exact_face_ref.GetAssembledOperator() == nullptr,
                  "IGR CPU L2 exact-face PA/QF uses operator path");

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
          dg_igr::dg_face::InteriorFaceSideDensityProjectionScratch
            solver_coeff_scratch;
          dg_igr::dg_face::ProjectInteriorFaceDensityCoefficient2D(
            state_fes_qf,
            solver_face_fqs,
            U_qf,
            /*alpha=*/0.3,
            mfem::dg_igr::IGRDefaults::kRhoFloor,
            solver_side_q,
            &solver_coeff_scratch);

          const double ip_sigma_solver =
            static_cast<double>(dg_igr::IGRIPKind::SIPG);
          const double kappa_solver =
            4.0 * (sigma_order_qf + 1) * (sigma_order_qf + 1);
          mfem::Vector exact_face_y(probe_local.Size());
          exact_face_y = 0.0;
          dg_igr::dg_face::InteriorFaceDGDiffusionApplyScratch
            solver_face_scratch;
          dg_igr::dg_face::AddInteriorFaceDGDiffusionApply2D(
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
            mfem::dg_igr::IGRDefaults::kRhoFloor);
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
          std::getenv("DG_IGR_REQUIRE_L2_EXACT_FACE_EQUIV") != nullptr;
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
    l2_solver.SetIPKind(dg_igr::IGRIPKind::SIPG);
    l2_solver.SetLinearSolver(dg_igr::IGRLinearSolverType::CG);
    l2_solver.SetPreconditioner(dg_igr::IGRPrecondType::Auto);
    l2_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(l2_solver.GetLastLinearConverged(),
              "IGR L2 SIPG CG solve converges");
    ctx.Check(l2_solver.GetLastLinearSolverName() == "cg",
              "IGR L2 SIPG explicit CG request stays on CG");

    // Guardrail: non-symmetric IP kind must not use CG.
    l2_solver.SetIPKind(dg_igr::IGRIPKind::NIPG);
    l2_solver.SetLinearSolver(dg_igr::IGRLinearSolverType::CG);
    l2_solver.SetPreconditioner(dg_igr::IGRPrecondType::Auto);
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
    l2_solver.SetIPKind(dg_igr::IGRIPKind::SIPG);

    // Guardrail: CG with ILU preconditioner on H1 should fallback preconditioner.
    h1_solver.SetLinearSolver(dg_igr::IGRLinearSolverType::CG);
    h1_solver.SetPreconditioner(dg_igr::IGRPrecondType::HypreILU);
    h1_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(h1_solver.GetLastPrecondName() == "boomeramg",
              "IGR guardrail falls back from ILU to BoomerAMG for CG");

    // Warm-start regression: warm-start should reduce iteration counts after step 0.
    dg_igr::IGRSolver cold_solver(sigma_l2_fes,
                                  state_fes_ws,
                                  dim_ws,
                                  num_eq_ws,
                                  /*alpha=*/0.5);
    dg_igr::IGRSolver warm_solver(sigma_l2_fes,
                                  state_fes_ws,
                                  dim_ws,
                                  num_eq_ws,
                                  /*alpha=*/0.5);
    for (dg_igr::IGRSolver* s : {&cold_solver, &warm_solver})
    {
      s->SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
      s->SetSolveMode(dg_igr::IGRSolveMode::Krylov);
      s->SetLinearSolver(dg_igr::IGRLinearSolverType::GMRES);
      s->SetPreconditioner(dg_igr::IGRPrecondType::None);
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

    dg_igr::IGRSolver igr_solver_s(sigma_fes_s,
                                   state_fes_s,
                                   dim_s,
                                   num_eq_s,
                                   /*alpha=*/0.25);
    igr_solver_s.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
    igr_solver_s.SetIPKind(dg_igr::IGRIPKind::SIPG);
    igr_solver_s.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
    igr_solver_s.SetLinearSolver(dg_igr::IGRLinearSolverType::Auto);
    igr_solver_s.SetPreconditioner(dg_igr::IGRPrecondType::Auto);
    igr_solver_s.SetLinearSolveParams(/*rtol=*/1.0e-12,
                                      /*atol=*/0.0,
                                      /*max_it=*/400,
                                      /*print=*/0);
    igr_solver_s.SetSigmaUpdatePolicy(dg_igr::IGRSigmaUpdateCadence::EveryStage,
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
    wrapped.BeginRKStep(/*step=*/0, /*t=*/0.0, /*dt=*/0.01);
    wrapped.SetRKStage(/*stage=*/0);

    mfem::Vector rhs(U_stage.Size());
    wrapped.Mult(U_stage, rhs);

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

    auto check_non_stage_fallback =
      [&](dg_igr::IGRSigmaUpdateCadence cadence,
          int interval,
          const std::string& label)
    {
      dg_igr::IGRSolver igr_solver_fb(sigma_fes_s,
                                      state_fes_s,
                                      dim_s,
                                      num_eq_s,
                                      /*alpha=*/0.25);
      igr_solver_fb.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
      igr_solver_fb.SetIPKind(dg_igr::IGRIPKind::SIPG);
      igr_solver_fb.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
      igr_solver_fb.SetLinearSolver(dg_igr::IGRLinearSolverType::Auto);
      igr_solver_fb.SetPreconditioner(dg_igr::IGRPrecondType::Auto);
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

    check_non_stage_fallback(dg_igr::IGRSigmaUpdateCadence::EveryStep,
                             /*interval=*/1,
                             "every-step");
    check_non_stage_fallback(dg_igr::IGRSigmaUpdateCadence::EveryNSteps,
                             /*interval=*/3,
                             "every-n-steps");
  }

  // Exact-face L2 PA/QF currently has no shared-face operator. Pin down the
  // boundary: MFEM exposes shared faces through FaceQuadratureSpace, and the
  // custom helpers must reject them instead of silently building a partial
  // MPI operator.
  if (mfem::Mpi::WorldSize() > 1)
  {
    constexpr int dim_shared = 2;
    mfem::Mesh mesh_shared =
      mfem::Mesh::MakeCartesian2D(/*nx=*/4,
                                  /*ny=*/2,
                                  mfem::Element::QUADRILATERAL,
                                  /*generate_edges=*/true,
                                  /*sx=*/1.0,
                                  /*sy=*/0.5);
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
      dg_igr::dg_face::SupportsInteriorFaceSideDensityProjection2D(
        shared_fes,
        shared_fqs,
        &side_reason);
    std::string trace_reason;
    const bool trace_supported =
      dg_igr::dg_face::SupportsInteriorFaceScalarTraceProjection2D(
        shared_fes,
        shared_fqs,
        &trace_reason);

    std::string shared_apply_reason;
    const bool shared_apply_supported =
      dg_igr::dg_face::SupportsSharedFaceDGDiffusionApplyData2D(
        shared_fes,
        shared_face_ir,
        &shared_apply_reason);

    int local_shared_apply_bad = 0;
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
        dg_igr::dg_face::SharedFaceDGDiffusionApplyScratch shared_scratch;
        const auto shared_info =
          dg_igr::dg_face::BuildSharedFaceDGDiffusionApplyData2D(
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
          static_cast<double>(dg_igr::IGRIPKind::SIPG);
        const double shared_kappa = 2.75;
        mfem::Vector shared_apply_y(shared_sigma.Size());
        shared_apply_y = 0.0;
        dg_igr::dg_face::SharedFaceDGDiffusionApplyScratch
          shared_apply_scratch;
        const auto shared_apply_info =
          dg_igr::dg_face::AddSharedFaceDGDiffusionApply2D(
            shared_fes,
            shared_face_ir,
            shared_sigma,
            shared_side_q,
            shared_ip_sigma,
            shared_kappa,
            shared_apply_y,
            &shared_apply_scratch);

        mfem::Vector shared_diag(shared_sigma.Size());
        shared_diag = 0.0;
        dg_igr::dg_face::AddSharedFaceDGDiffusionLocalDiagonal2D(
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
      dg_igr::IGRSolver shared_exact_solver(shared_fes,
                                            shared_state_fes,
                                            dim_shared,
                                            num_eq_shared,
                                            /*alpha=*/0.19);
      shared_exact_solver.SetLinearSolver(dg_igr::IGRLinearSolverType::CG);
      shared_exact_solver.SetPreconditioner(dg_igr::IGRPrecondType::None);
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
          local_exact_operator_shared_nf != local_pmesh_shared ||
          local_exact_operator_local_nf <= 0)
      {
        local_exact_operator_bad = 1;
      }
    }

    const int local_bad_support =
      (local_shared_in_fqs > 0 &&
       (!side_supported || !trace_supported)) ? 1 : 0;
    int global_bad_support = 0;
    int global_shared_apply_bad = 0;
    int global_shared_apply_nf = 0;
    int global_exact_operator_bad = 0;
    int global_exact_operator_shared_nf = 0;
    int global_exact_operator_local_nf = 0;
    int global_exact_operator_converged = 0;
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
    ctx.AddParity(static_cast<double>(global_shared_in_fqs +
                                      global_pmesh_shared +
                                      global_bad_support +
                                      global_shared_apply_bad +
                                      global_shared_apply_nf +
                                      global_exact_operator_bad +
                                      global_exact_operator_shared_nf));
    ctx.Check(global_pmesh_shared > 0,
              "IGR exact-face MPI guard test creates shared ParMesh faces");
    ctx.Check(global_shared_in_fqs > 0,
              "IGR exact-face MPI guard sees shared faces in FaceQuadratureSpace");
    ctx.Check(global_bad_support == 0,
              "IGR exact-face local helpers support parallel face spaces");
    ctx.Check(global_shared_apply_bad == 0,
              "IGR exact-face shared-face metadata builder is valid");
    ctx.Check(global_shared_apply_nf == global_pmesh_shared,
              "IGR exact-face shared-face metadata covers ParMesh shared faces");
    ctx.Check(global_exact_operator_bad == 0,
              "IGR exact-face MPI operator path converges with shared faces");
    ctx.Check(global_exact_operator_shared_nf == global_pmesh_shared,
              "IGR exact-face MPI operator constructs shared face data");
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

    dg_igr::IGRSolver solver_q(sigma_fes_q,
                               state_fes_q,
                               dim_q,
                               num_eq_q,
                               /*alpha=*/0.5);
    solver_q.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
    solver_q.SetLinearSolver(dg_igr::IGRLinearSolverType::Auto);
    solver_q.SetPreconditioner(dg_igr::IGRPrecondType::Auto);
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

} // namespace dg_igr_tests

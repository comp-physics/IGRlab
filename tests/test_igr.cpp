#include "physics/euler/conversions.hpp"
#include "physics/igr/IGR.hpp"
#include "physics/igr/igr_flux.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/riemann/side_scope_utils.hpp"

#include <algorithm>
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
      orig_mesh, orig_mesh.CreatePeriodicVertexMapping(translations));
}

mfem::Mesh MakePeriodicMesh2D(int nx, int ny, double sx = 1.0, double sy = 1.0)
{
  mfem::Mesh orig_mesh = mfem::Mesh::MakeCartesian2D(
      nx, ny, mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true, sx, sy);
  mfem::Vector shift_x(2);
  shift_x(0) = sx;
  shift_x(1) = 0.0;
  mfem::Vector shift_y(2);
  shift_y(0) = 0.0;
  shift_y(1) = sy;
  std::vector<mfem::Vector> translations = {shift_x, shift_y};
  return mfem::Mesh::MakePeriodic(
      orig_mesh, orig_mesh.CreatePeriodicVertexMapping(translations));
}

mfem::FaceElementTransformations& GetFaceTr(mfem::ParMesh& pmesh, int face)
{
  mfem::FaceElementTransformations* Tr = pmesh.GetFaceElementTransformations(face);
  MFEM_VERIFY(Tr != nullptr, "Failed to obtain face transformations.");
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(Tr->GetGeometryType(), /*order=*/1);
  const mfem::IntegrationPoint& ip = ir.IntPoint(0);
  Tr->SetAllIntPoints(&ip);
  return *Tr;
}

class StepSigmaCoefficient final : public mfem::Coefficient
{
public:
  StepSigmaCoefficient(double left, double right, double split_x)
    : left_(left), right_(right), split_x_(split_x)
  {}

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

class LinearAverageFlux final : public mfem::FluxFunction
{
public:
  LinearAverageFlux(int dim, const mfem::Vector &dir)
    : mfem::FluxFunction(dim + 2, dim), dir_(dir)
  {
    MFEM_VERIFY(dir_.Size() == dim,
                "LinearAverageFlux direction dimension mismatch.");
  }

  mfem::real_t ComputeFlux(const mfem::Vector &state,
                           mfem::ElementTransformation &,
                           mfem::DenseMatrix &flux) const override
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

  mfem::real_t ComputeAvgFlux(const mfem::Vector &state1,
                              const mfem::Vector &state2,
                              mfem::ElementTransformation &Tr,
                              mfem::DenseMatrix &flux) const override
  {
    mfem::Vector avg(state1);
    avg += state2;
    avg *= 0.5;
    return ComputeFlux(avg, Tr, flux);
  }

private:
  mfem::Vector dir_;
};

double DotMomentumNormal(const mfem::Vector& cons, const mfem::Vector& normal, int dim)
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
  return dmn * inv_drho +
         (mn1 * drho - dmn * rho1) * inv_drho2 * log_ratio;
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
  MPI_Allreduce(&local, &global, 1, mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM, comm);
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

SPDProbeSummary ProbeSPD(const mfem::HypreParMatrix& A,
                         MPI_Comm comm,
                         int samples = 4)
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

} // namespace

int RunIGRTests(TestContext& ctx)
{
  constexpr int dim = 2;
  constexpr double gamma = 1.4;
  const double tol = 1.0e-11;

  mfem::EulerFlux base_flux(dim, gamma);
  mfem::FluxWithIGR igr_flux(base_flux);

  const mfem::Vector state = EulerConsFromPrim2D(
      /*rho=*/1.2, /*u=*/2.0, /*v=*/-0.5, /*p=*/1.0, gamma);
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
      ctx.CheckNear(igr_fn(i), base_fn(i), tol,
                    "FluxWithIGR matches base flux when sigma is null, component " +
                        std::to_string(i));
    }
  }

  // --- With constant sigma, momentum/energy offsets match the documented form ---
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      /*nx=*/2, /*ny=*/2, mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true, /*sx=*/1.0, /*sy=*/1.0);
  mfem::ParMesh pmesh(MPI_COMM_SELF, mesh);
  MFEM_VERIFY(pmesh.GetNumFaces() > 0, "IGR flux tests require a mesh with faces.");
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
  ctx.CheckNear(igr_fn(1) - base_fn(1), sigma_value * normal(0), tol,
                "FluxWithIGR adds sigma*n_x to momentum_x");
  ctx.CheckNear(igr_fn(2) - base_fn(2), sigma_value * normal(1), tol,
                "FluxWithIGR adds sigma*n_y to momentum_y");
  ctx.CheckNear(igr_fn(3) - base_fn(3), sigma_value * un, tol,
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
      ctx.CheckNear(igr_F(1 + d, d) - base_F(1 + d, d), sigma_value, tol,
                    "FluxWithIGR adds +sigma to momentum tensor diagonal, dim=" +
                        std::to_string(d));
      const double expected_energy = sigma_value * state(1 + d) / rho_vol;
      ctx.CheckNear(igr_F(1 + dim, d) - base_F(1 + dim, d),
                    expected_energy, tol,
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
        /*rho=*/0.9, /*u=*/1.8, /*v=*/-0.3, /*p=*/1.0, gamma);
    const mfem::Vector state_avg_2 = EulerConsFromPrim2D(
        /*rho=*/1.4, /*u=*/-0.2, /*v=*/0.6, /*p=*/0.9, gamma);

    mfem::Vector base_avg_fn(dim + 2);
    mfem::Vector igr_avg_fn(dim + 2);
    linear_base.ComputeAvgFluxDotN(state_avg_1, state_avg_2, normal, Tr_face,
                                    base_avg_fn);
    igr_avg.ComputeAvgFluxDotNAtSide(state_avg_1, state_avg_2, normal, Tr_face,
                                     mfem::FaceSide::Left, igr_avg_fn);

    mfem::Vector expected_avg(base_avg_fn);
    for (int d = 0; d < dim; ++d)
    {
      expected_avg(1 + d) += sigma_value * normal(d);
    }
    expected_avg(1 + dim) += sigma_value *
                             ExactAvgUnOverRho(state_avg_1, state_avg_2,
                                               normal, dim);

    for (int i = 0; i < expected_avg.Size(); ++i)
    {
      ctx.CheckNear(igr_avg_fn(i), expected_avg(i), 5.0e-12,
                    "FluxWithIGR exact averaged sigma correction, component " +
                        std::to_string(i));
    }
  }

  // --- Interior-face side selection: sigma should come from the matching side ---
  {
    mfem::Mesh mesh_if = mfem::Mesh::MakeCartesian2D(
        /*nx=*/4, /*ny=*/2, mfem::Element::QUADRILATERAL,
        /*generate_edges=*/true, /*sx=*/1.0, /*sy=*/1.0);
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
        /*rho=*/1.0, /*u=*/1.5, /*v=*/0.25, /*p=*/1.0, gamma);
    const mfem::Vector state_right = EulerConsFromPrim2D(
        /*rho=*/0.8, /*u=*/-0.75, /*v=*/0.5, /*p=*/0.9, gamma);
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
      mfem::FaceElementTransformations* Tr = pmesh_if.GetFaceElementTransformations(f);
      if (!Tr || !Tr->Elem2) { continue; }
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
        expected(i) = 0.5 * (scaled_max_speed * (state_left(i) - state_right(i)) +
                             fluxN1(i) + fluxN2(i));
      }

      mfem::Vector actual(dim + 2);
      num_flux.Eval(state_left, state_right, normal_if, *Tr, actual);

      for (int i = 0; i < actual.Size(); ++i)
      {
        local_max_err = std::max(local_max_err, std::abs(actual(i) - expected(i)));
      }
      ++local_face_count;
    }

    const double global_faces = GlobalSum(static_cast<double>(local_face_count));
    ctx.Check(global_faces > 0.0, "IGR interior-face test found interior faces");

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
        /*nx=*/2, /*ny=*/1, mfem::Element::QUADRILATERAL,
        /*generate_edges=*/true, /*sx=*/1.0, /*sy=*/1.0);
    mfem::ParMesh pmesh_side(MPI_COMM_SELF, mesh_side);

    mfem::DG_FECollection fec_side(/*order=*/0, dim);
    mfem::ParFiniteElementSpace sigma_fes_side(&pmesh_side, &fec_side, /*vdim=*/1);
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
      mfem::FaceElementTransformations* Tr = pmesh_side.GetFaceElementTransformations(f);
      if (Tr && Tr->Elem1 && Tr->Elem2)
      {
        interior_face = f;
        break;
      }
    }
    MFEM_VERIFY(interior_face >= 0,
                "Deterministic side-selection stress test requires an interior face.");

    mfem::FaceElementTransformations* Tr = pmesh_side.GetFaceElementTransformations(interior_face);
    MFEM_VERIFY(Tr != nullptr, "Failed to obtain interior face transformations.");
    const mfem::IntegrationRule& ir_face =
        mfem::IntRules.Get(Tr->GetGeometryType(), /*order=*/1);
    const mfem::IntegrationPoint& ip_face = ir_face.IntPoint(0);
    Tr->SetAllIntPoints(&ip_face);

    mfem::Vector state = EulerConsFromPrim2D(
        /*rho=*/1.1, /*u=*/0.9, /*v=*/-0.3, /*p=*/1.0, gamma);
    mfem::Vector normal(dim);
    mfem::CalcOrtho(Tr->Jacobian(), normal);

    mfem::Vector left_ref(state.Size());
    mfem::Vector right_ref(state.Size());
    const mfem::real_t left_speed_ref = mfem::riemann_detail::ComputeFluxDotNForSide(
        igr_flux_side, mfem::FaceSide::Left, state, normal, *Tr, left_ref);
    const mfem::real_t right_speed_ref = mfem::riemann_detail::ComputeFluxDotNForSide(
        igr_flux_side, mfem::FaceSide::Right, state, normal, *Tr, right_ref);

    // Side regression snapshot: left/right must differ when sigma differs per element.
    const double ref_diff =
        std::abs(left_ref(1) - right_ref(1)) +
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
      const mfem::real_t speed_explicit = mfem::riemann_detail::ComputeFluxDotNForSide(
          igr_flux_side, side, state, normal, *Tr, flux_explicit);

      const mfem::Vector& flux_ref =
          (side == mfem::FaceSide::Left) ? left_ref : right_ref;
      const mfem::real_t speed_ref =
          (side == mfem::FaceSide::Left) ? left_speed_ref : right_speed_ref;

      for (int i = 0; i < flux_ref.Size(); ++i)
      {
        ctx.CheckNear(flux_explicit(i), flux_ref(i), tol,
                      "Explicit-side flux remains deterministic, iter=" +
                          std::to_string(iter) + ", component=" + std::to_string(i));
      }
      ctx.CheckNear(speed_explicit, speed_ref, tol,
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

    mfem::DG_FECollection state_fec(/*order=*/2, dim1, mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace vfes(&pmesh_dg, &state_fec, num_eq1,
                                     mfem::Ordering::byNODES);
    mfem::ParGridFunction U_const(&vfes);
    const mfem::Vector cons_const =
        EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/0.0, /*p=*/1.0, gamma);
    FillConstantState(cons_const, U_const);

    mfem::DG_FECollection sigma_fec_dg(/*order=*/1, dim1,
                                       mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace sigma_fes_dg(&pmesh_dg, &sigma_fec_dg,
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
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form_integrator),
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
        /*nx=*/2, /*ny=*/2, mfem::Element::QUADRILATERAL,
        /*generate_edges=*/true, /*sx=*/1.0, /*sy=*/1.0);
    mfem::ParMesh pmesh_iso(MPI_COMM_SELF, mesh_iso);

    mfem::DG_FECollection sigma_fec(/*order=*/1, igr_dim);
    mfem::DG_FECollection state_fec(/*order=*/1, igr_dim);
    mfem::ParFiniteElementSpace sigma_fes(&pmesh_iso, &sigma_fec, /*vdim=*/1);
    mfem::ParFiniteElementSpace state_fes(&pmesh_iso, &state_fec, igr_num_eq,
                                          mfem::Ordering::byNODES);

    mfem::ParGridFunction U(&state_fes);
    mfem::VectorFunctionCoefficient U_coeff(
        igr_num_eq,
        [&](const mfem::Vector& x, mfem::Vector& cons)
        {
          mfem::Vector prim(igr_num_eq);
          prim(0) = 1.0 + 0.1 * x(0);               // rho
          prim(1) = 1.5 + 0.2 * x(1);               // u
          prim(2) = -0.25 + 0.1 * x(0);             // v
          prim(1 + igr_dim) = 1.0 + 0.05 * x(1);    // p
          cons.SetSize(igr_num_eq);
          mfem::EulerPrimToConservative(prim, igr_dim, gamma, cons);
        });
    U.ProjectCoefficient(U_coeff);
    U.ExchangeFaceNbrData();

    dg_igr::IGRSolver solver_a(sigma_fes, state_fes, igr_dim, igr_num_eq, /*alpha=*/0.05);
    dg_igr::IGRSolver solver_b(sigma_fes, state_fes, igr_dim, igr_num_eq, /*alpha=*/0.35);
    solver_a.SetRhoFloor(1.0e-12);
    solver_b.SetRhoFloor(5.0e-2);
    solver_a.SetLinearSolveParams(/*rtol=*/1.0e-10, /*atol=*/1.0e-14, /*max_it=*/200, /*print=*/0);
    solver_b.SetLinearSolveParams(/*rtol=*/1.0e-10, /*atol=*/1.0e-14, /*max_it=*/200, /*print=*/0);

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
    ctx.Check(repeat_err < 1.0e-11,
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

    mfem::DG_FECollection state_fec_ws(/*order=*/2, dim_ws,
                                       mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace state_fes_ws(&pmesh_ws, &state_fec_ws, num_eq_ws,
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
    mfem::ParFiniteElementSpace sigma_h1_fes(&pmesh_ws, &sigma_h1_fec, /*vdim=*/1);
    dg_igr::IGRSolver h1_solver(sigma_h1_fes, state_fes_ws, dim_ws, num_eq_ws,
                                /*alpha=*/0.5);
    h1_solver.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
    h1_solver.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
    h1_solver.SetLinearSolver(dg_igr::IGRLinearSolverType::Auto);
    h1_solver.SetPreconditioner(dg_igr::IGRPrecondType::Auto);
    h1_solver.SetLinearSolveParams(/*rtol=*/1.0e-10, /*atol=*/0.0,
                                   /*max_it=*/250, /*print=*/0);
    project_state(/*phase=*/0.0);
    h1_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(h1_solver.GetLastLinearConverged(),
              "IGR H1 auto policy converges");
    ctx.Check(h1_solver.GetLastLinearSolverName() == "cg",
              "IGR H1 auto policy selects CG");
    ctx.Check(h1_solver.GetLastPrecondName() == "boomeramg",
              "IGR H1 auto policy selects BoomerAMG");

    // L2 auto policy with SIPG should be SPD-by-construction -> CG + BoomerAMG.
    mfem::DG_FECollection sigma_l2_fec(/*order=*/1, dim_ws,
                                       mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace sigma_l2_fes(&pmesh_ws, &sigma_l2_fec, /*vdim=*/1,
                                             mfem::Ordering::byNODES);
    dg_igr::IGRSolver l2_solver(sigma_l2_fes, state_fes_ws, dim_ws, num_eq_ws,
                                /*alpha=*/0.5);
    l2_solver.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
    l2_solver.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
    l2_solver.SetLinearSolver(dg_igr::IGRLinearSolverType::Auto);
    l2_solver.SetPreconditioner(dg_igr::IGRPrecondType::Auto);
    l2_solver.SetLinearSolveParams(/*rtol=*/1.0e-10, /*atol=*/0.0,
                                   /*max_it=*/250, /*print=*/0);
    project_state(/*phase=*/0.0);
    l2_solver.UpdateFromState(U_ws, /*warm_start=*/false);
    ctx.Check(l2_solver.GetLastLinearConverged(),
              "IGR L2 auto policy converges");
    ctx.Check(l2_solver.GetLastLinearSolverName() == "cg",
              "IGR L2 SIPG auto policy selects CG");
    ctx.Check(l2_solver.GetLastPrecondName() == "boomeramg",
              "IGR L2 auto policy selects BoomerAMG");

    const mfem::HypreParMatrix* A_l2_sipg = l2_solver.GetAssembledOperator();
    MFEM_VERIFY(A_l2_sipg != nullptr,
                "IGR L2 SIPG SPD probe requires assembled Hypre operator.");
    const SPDProbeSummary sipg_probe = ProbeSPD(*A_l2_sipg, MPI_COMM_SELF);
    ctx.LogMetric("igr_l2_sipg_spd_probe max_sym_rel=" +
                  FormatSci(sipg_probe.max_sym_rel) +
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
    dg_igr::IGRSolver cold_solver(sigma_l2_fes, state_fes_ws, dim_ws, num_eq_ws,
                                  /*alpha=*/0.5);
    dg_igr::IGRSolver warm_solver(sigma_l2_fes, state_fes_ws, dim_ws, num_eq_ws,
                                  /*alpha=*/0.5);
    for (dg_igr::IGRSolver* s : {&cold_solver, &warm_solver})
    {
      s->SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
      s->SetSolveMode(dg_igr::IGRSolveMode::Krylov);
      s->SetLinearSolver(dg_igr::IGRLinearSolverType::GMRES);
      s->SetPreconditioner(dg_igr::IGRPrecondType::None);
      s->SetLinearSolveParams(/*rtol=*/1.0e-12, /*atol=*/0.0,
                              /*max_it=*/500, /*print=*/0);
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
    ctx.LogMetric("igr_warm_start_same_rhs cold_iters=" +
                  FormatSci(cold_replay_iters) +
                  " warm_iters=" + FormatSci(warm_replay_iters));
    ctx.Check(warm_replay_iters <= cold_replay_iters,
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

      cold_iters.push_back(static_cast<double>(cold_solver.GetLastLinearIters()));
      warm_iters.push_back(static_cast<double>(warm_solver.GetLastLinearIters()));
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
    ctx.Check(warm_avg <= cold_avg,
              "IGR warm-start average iteration count is not worse than cold-start");
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

    mfem::DG_FECollection state_fec_q(/*order=*/2, dim_q,
                                      mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace state_fes_q(&pmesh_q, &state_fec_q, num_eq_q,
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

    mfem::DG_FECollection sigma_fec_q(/*order=*/1, dim_q,
                                      mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace sigma_fes_q(&pmesh_q, &sigma_fec_q, /*vdim=*/1,
                                            mfem::Ordering::byNODES);

    dg_igr::IGRSolver solver_q(sigma_fes_q, state_fes_q, dim_q, num_eq_q,
                               /*alpha=*/0.5);
    solver_q.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
    solver_q.SetLinearSolver(dg_igr::IGRLinearSolverType::Auto);
    solver_q.SetPreconditioner(dg_igr::IGRPrecondType::Auto);
    solver_q.SetLinearSolveParams(/*rtol=*/1.0e-8, /*atol=*/0.0,
                                  /*max_it=*/500, /*print=*/0);
    solver_q.UpdateFromState(U_q, /*warm_start=*/false);

    ctx.Check(solver_q.GetLastLinearConverged(),
              "IGR quasi-1D L2 solve converges");

    const mfem::ParGridFunction& sigma_q = solver_q.GetSigma();
    const mfem::ParFiniteElementSpace* sfes_q = sigma_q.ParFESpace();
    MFEM_VERIFY(sfes_q != nullptr, "IGR quasi-1D test requires sigma FE space.");

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

    MPI_Allreduce(local_col_min.data(), global_col_min.data(), nx_q,
                  mfem::MPITypeMap<double>::mpi_type, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_col_max.data(), global_col_max.data(), nx_q,
                  mfem::MPITypeMap<double>::mpi_type, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local_col_sum.data(), global_col_sum.data(), nx_q,
                  mfem::MPITypeMap<double>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_col_count.data(), global_col_count.data(), nx_q,
                  mfem::MPITypeMap<double>::mpi_type, MPI_SUM, MPI_COMM_WORLD);

    double max_y_spread = 0.0;
    double x_min_avg = std::numeric_limits<double>::infinity();
    double x_max_avg = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < nx_q; ++i)
    {
      if (global_col_count[i] <= 0.0) { continue; }
      max_y_spread = std::max(max_y_spread, global_col_max[i] - global_col_min[i]);
      const double avg = global_col_sum[i] / global_col_count[i];
      x_min_avg = std::min(x_min_avg, avg);
      x_max_avg = std::max(x_max_avg, avg);
    }
    const double x_span = x_max_avg - x_min_avg;
    const double y_to_x_ratio =
        max_y_spread / std::max(x_span, 1.0e-14);
    ctx.LogMetric("igr_quasi1d_sigma_spread max_y_spread=" +
                  FormatSci(max_y_spread) +
                  " x_span=" + FormatSci(x_span) +
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

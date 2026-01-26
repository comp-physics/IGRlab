#include "physics/igr/igr_flux.hpp"
#include "numerics/riemann/rusanov.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "test_utils.hpp"

namespace dg_igr_tests
{

namespace
{

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

double DotMomentumNormal(const mfem::Vector& cons, const mfem::Vector& normal, int dim)
{
  double un = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    un += cons(1 + d) * normal(d);
  }
  return un;
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
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  if (pmesh.GetNFaces() == 0)
  {
    return ctx.failures;
  }
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

  // --- Interior-face side selection: sigma should come from the matching side ---
  {
    mfem::Mesh mesh_if = mfem::Mesh::MakeCartesian2D(
        /*nx=*/4, /*ny=*/2, mfem::Element::QUADRILATERAL,
        /*generate_edges=*/true, /*sx=*/1.0, /*sy=*/1.0);
    mfem::ParMesh pmesh_if(MPI_COMM_WORLD, mesh_if);

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

    for (int f = 0; f < pmesh_if.GetNFaces(); ++f)
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

  return ctx.failures;
}

} // namespace dg_igr_tests

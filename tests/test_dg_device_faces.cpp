#include "igrlab/environment.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "physics/igr/igr_flux.hpp"
#include "../src/numerics/dg/DGEntropyFaceKernels.hpp"
#include "test_utils.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <vector>

namespace
{
using mfem::real_t;
using dg_igr_tests::TestContext;

void NearVector(TestContext& ctx, const mfem::Vector& actual,
                const mfem::Vector& expected, double tol, const std::string& label)
{
  mfem::Vector error(actual); error -= expected;
  const double norm = error.Normlinf();
  ctx.Check(std::isfinite(norm) && norm <= tol * std::max(real_t(1.0), expected.Normlinf()),
            label + " max error=" + dg_igr_tests::FormatSci(norm));
}

void AlgebraOracle(TestContext& ctx)
{
  for (double gamma : {1.2, 1.4, 1.67})
    for (double jump : {0.0, 1.e-13, 0.04, 0.3})
      for (double sigma : {0.0, -0.1, 0.2})
        for (const auto& n : std::vector<std::array<double, 2>>{
               {1.0, 0.0}, {0.0, -0.7}, {-0.3, 0.8}, {0.4, 0.6}})
        {
          real_t l[4] = {1.1, 0.31, -0.22, 3.8};
          real_t r[4] = {1.1 + jump, 0.31 - jump, -0.22 + 0.2 * jump, 3.8 + jump};
          real_t nd[2] = {n[0], n[1]}, packed[4];
          mfem::Vector ul(l, 4), ur(r, 4), normal(nd, 2), oracle, diss;
          // This oracle retains the production Vector implementation and its
          // directional flux/Winters helpers; it does not call the raw helper.
          mfem::dgsem::EntropyConservativeFluxDotN(ul, ur, normal, gamma,
            mfem::dgsem::EntropyFluxType::Ranocha, oracle, sigma, -0.3 * sigma);
          mfem::dgsem::WintersDissipation(ul, ur, normal, gamma, diss, sigma, -0.3 * sigma);
          oracle += diss;
          ctx.Check(mfem::dg_igr_detail::packed_entropy::Flux(
              l, r, nd, gamma, sigma, -0.3 * sigma, packed), "admissible raw ES flux");
          mfem::Vector result(packed, 4);
          NearVector(ctx, result, oracle, 8.e-14, "raw Ranocha/Winters oracle");
          mfem::EulerFlux base(2, gamma);
          mfem::FaceElementTransformations tr;
          mfem::Vector scratch(4);
          ctx.CheckNear(mfem::dg_igr_detail::packed_entropy::EulerSpeed(l, nd, gamma),
                        base.ComputeFluxDotN(ul, normal, tr, scratch), 2.e-14,
                        "thermal Euler speed oracle");
        }
}

void Fill(mfem::ParFiniteElementSpace& fes, mfem::Vector& state,
          mfem::ParGridFunction& sigma, double sigma_scale)
{
  state.SetSize(fes.GetVSize());
  auto* values = state.HostWrite();
  auto* sig = sigma.HostWrite();
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs, sdofs;
    fes.GetElementDofs(e, dofs);
    sigma.ParFESpace()->GetElementDofs(e, sdofs);
    const auto& nodes = fes.GetFE(e)->GetNodes();
    for (int i = 0; i < dofs.Size(); ++i)
    {
      const auto& ip = nodes.IntPoint(i);
      const double rho = 1.0 + 0.013 * e + 0.03 * ip.x * ip.y;
      const double vx = 0.24 + 0.02 * ip.x - 0.004 * e;
      const double vy = -0.12 + 0.01 * ip.y + 0.003 * e;
      const double pressure = 1.0 + 0.02 * ip.x + 0.003 * e;
      const real_t u[4] = {rho, rho * vx, rho * vy,
        pressure / 0.4 + 0.5 * rho * (vx * vx + vy * vy)};
      for (int c = 0; c < 4; ++c) { values[fes.DofToVDof(dofs[i], c)] = u[c]; }
      sig[sdofs[i]] = sigma_scale * (0.03 + 0.01 * e + 0.02 * ip.x * ip.y);
    }
  }
}

std::string Apply(mfem::DGHyperbolicConservationLaws& op,
                  const mfem::Vector& state, mfem::Vector& rhs)
{
  std::ostringstream output;
  auto* previous = std::cout.rdbuf(output.rdbuf());
  op.Mult(state, rhs);
  std::cout.rdbuf(previous);
  return output.str();
}

void OperatorOracle(TestContext& ctx, int basis, int ordering, int order,
                    int mutation = 0)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
    3, 3, mfem::Element::QUADRILATERAL, false, 1.3, 0.9, false);
  // Shear the domain: exercises oblique normals and periodic face orientation.
  for (int v = 0; v < original.GetNV(); ++v)
    original.GetVertex(v)[0] += 0.2 * original.GetVertex(v)[1];
  std::vector<mfem::Vector> translations = {
    mfem::Vector({1.3, 0.0}), mfem::Vector({0.18, 0.9})};
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
    original, original.CreatePeriodicVertexMapping(translations));
  mfem::ParMesh mesh(MPI_COMM_SELF, periodic);
  mfem::L2_FECollection fec(order, 2, basis);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, 4, ordering);
  mfem::ParFiniteElementSpace sfes(&mesh, &fec);
  mfem::ParGridFunction sigma(&sfes);
  mfem::Vector state, actual(fes.GetVSize()), expected(fes.GetVSize()), repeat(fes.GetVSize());
  Fill(fes, state, sigma, 1.0);
  mfem::EulerFlux base(2, 1.4);
  mfem::FluxWithIGR physical(base, 1.4);
  physical.SetSigmaField(&sigma);
  mfem::DGSEMSurfaceFlux flux(physical);
  flux.SetGamma(1.4);
  flux.SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::EntropyStableWinters);
  flux.SetEntropyFluxType(mfem::dgsem::EntropyFluxType::Ranocha);
  mfem::DGSEMSurfaceFlux replacement(physical);
  replacement.SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::EntropyStableWinters);
  mfem::DGHyperbolicConservationLaws candidate(fes, flux, 2, true);
  // The owning constructor deliberately retains the MFEM NonlinearForm route.
  mfem::DGHyperbolicConservationLaws oracle(fes,
    std::make_unique<mfem::HyperbolicFormIntegrator>(flux, 2), true);
  candidate.SetDGSEMGamma(1.4); oracle.SetDGSEMGamma(1.4);
  oracle.SetNumericalFlux(&flux);
  if (mutation == 1) { candidate.SetFaceQuadratureOrderOffset(5); }
  if (mutation == 2) { physical.SetSigmaAwareCFL(true); }
  if (mutation == 3) { flux.SetEntropyFluxType(mfem::dgsem::EntropyFluxType::Chandrashekar); }
  if (mutation == 4) { physical.SetSigmaField(nullptr); }
  if (mutation == 5) { candidate.SetNumericalFlux(&replacement); }
  const std::string log = Apply(candidate, state, actual);
  Apply(oracle, state, expected);
  const bool admitted = order == 5 && (mutation == 0 || mutation == 4);
  if (mfem::Device::Allows(mfem::Backend::CUDA_MASK))
  {
    ctx.Check(log.find(admitted ? "selected=packed-mfem-forall-euler-es-winters"
                               : "selected=nonlinearForm->Mult") != std::string::npos,
              "p5 qualification and policy fallback select the expected face route");
  }
  NearVector(ctx, actual, expected, 2.e-11, "full DG RHS versus MFEM face oracle");
  ctx.CheckNear(candidate.GetMaxCharSpeed(), oracle.GetMaxCharSpeed(), 2.e-12,
                "full DG maximum speed versus MFEM oracle");
  Apply(candidate, state, repeat);
  NearVector(ctx, repeat, actual, admitted ? 0.0 : 2.e-13,
             "owned face gather repeats deterministically");
  // Current sigma values must be read every application, not cached with shape.
  Fill(fes, state, sigma, 1.7);
  Apply(candidate, state, actual); Apply(oracle, state, expected);
  NearVector(ctx, actual, expected, 2.e-11, "changed sigma values versus oracle");
  if (admitted)
  {
    // Mutable speed policy must return to the host route even after packing.
    physical.SetSigmaAwareCFL(true);
    Apply(candidate, state, actual); Apply(oracle, state, expected);
    NearVector(ctx, actual, expected, 2.e-11, "runtime speed-policy fallback");
    physical.SetSigmaAwareCFL(false);
    Apply(candidate, state, actual); Apply(oracle, state, expected);
    NearVector(ctx, actual, expected, 2.e-11, "runtime policy restores exact ES path");
  }
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device(argc > 1 ? argv[1] : "cpu");
  unsetenv("DG_IGR_DG_FORCE_NLFORM_FACE");
  TestContext ctx;
  AlgebraOracle(ctx);
  for (int basis : {mfem::BasisType::GaussLegendre, mfem::BasisType::GaussLobatto})
    for (int ordering : {mfem::Ordering::byNODES, mfem::Ordering::byVDIM})
      OperatorOracle(ctx, basis, ordering, 5);
  OperatorOracle(ctx, mfem::BasisType::GaussLobatto, mfem::Ordering::byNODES, 3);
  for (int mutation = 1; mutation <= 5; ++mutation)
    OperatorOracle(ctx, mfem::BasisType::GaussLobatto, mfem::Ordering::byNODES, 5, mutation);
  return ctx.failures == 0 ? 0 : 1;
}

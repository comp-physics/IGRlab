#pragma once

#include "igrlab/environment.hpp"

#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_flux.hpp"
#include "numerics/dgsem/dgsem_reference.hpp"
#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "physics/euler/conversions.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace dg_igr_tests
{

inline void RunDGSEMDeterministicFluxFixtures(TestContext& ctx)
{
  const double tol =
    mfem::dgsem::DGSEMReferenceTolerances::pointwise_fixture;

  // Frozen p2 scalar flux-differencing fixture.  On [0,1], p2 LGL has
  // Dsplit off-diagonals [[0,8,-2],[-2,0,2],[2,-8,0]].  With states
  // [1,2,4] and symmetric arithmetic flux, the exact action is [7,3,-19].
  const auto& basis = mfem::dgsem::GetLGLBasis1D(/*order=*/2);
  mfem::DenseMatrix state(/*height=*/3, /*width=*/1);
  state(0, 0) = 1.0;
  state(1, 0) = 2.0;
  state(2, 0) = 4.0;
  const mfem::dgsem::VolumeFlux arithmetic_flux =
    [](const mfem::Vector& left,
       const mfem::Vector& right,
       int,
       mfem::Vector& out)
  {
    out.SetSize(1);
    out(0) = 0.5 * (left(0) + right(0));
  };
  mfem::DenseMatrix action;
  mfem::dgsem::FluxDifferencing1D(
    state, basis.Dsplit, arithmetic_flux, action);
  constexpr double expected_volume[] = {7.0, 3.0, -19.0};
  for (int i = 0; i < 3; ++i)
  {
    ctx.CheckNear(action(i, 0),
                  expected_volume[i],
                  tol,
                  "DGSEM frozen p2 volume action entry " +
                    std::to_string(i));
  }

  // Frozen equal-state Euler/Rusanov surface fixture with scaled normal n=2.
  // U=[1,0.25,2.53125] gives F.n=[0.5,2.125,1.765625].
  constexpr double gamma = 1.4;
  mfem::Vector primitive(3);
  primitive(0) = 1.0;
  primitive(1) = 0.25;
  primitive(2) = 1.0;
  mfem::Vector conservative(3);
  mfem::EulerPrimToConservative(primitive, /*dim=*/1, gamma, conservative);

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(/*n=*/1, /*sx=*/1.0);
  mfem::FaceElementTransformations* tr = mesh.GetBdrFaceTransformations(0);
  MFEM_VERIFY(tr, "DGSEM surface fixture requires a boundary transform.");
  const mfem::IntegrationPoint& ip =
    mfem::IntRules.Get(mfem::Geometry::POINT, 1).IntPoint(0);
  tr->SetAllIntPoints(&ip);

  mfem::EulerFlux euler_flux(/*dim=*/1, gamma);
  mfem::DGSEMSurfaceFlux surface_flux(euler_flux);
  surface_flux.SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::Rusanov);
  surface_flux.SetGamma(gamma);
  mfem::Vector normal(1);
  normal(0) = 2.0;
  mfem::Vector numerical_flux(3);
  surface_flux.Eval(
    conservative, conservative, normal, *tr, numerical_flux);
  constexpr double expected_surface[] = {0.5, 2.125, 1.765625};
  for (int eq = 0; eq < 3; ++eq)
  {
    ctx.CheckNear(numerical_flux(eq),
                  expected_surface[eq],
                  tol,
                  "DGSEM frozen equal-state surface flux entry " +
                    std::to_string(eq));
  }

  const char* output_dir = ::igrlab::GetEnv("DG_IGR_DGSEM_REFERENCE_OUTPUT_DIR");
  if (output_dir && mfem::Mpi::Root())
  {
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    std::ofstream out(std::filesystem::path(output_dir) /
                      "deterministic_flux_fixtures.json");
    out << "{\n"
        << "  \"schema\": \"dgsem-deterministic-flux-fixtures-v1\",\n"
        << "  \"volume\": {\"order\": 2, \"state\": [1,2,4], "
           "\"expected_action\": [7,3,-19], \"tolerance\": "
        << tol << "},\n"
        << "  \"surface\": {\"state\": [1,0.25,2.53125], "
           "\"normal\": [2], \"expected_flux\": "
           "[0.5,2.125,1.765625], \"tolerance\": "
        << tol << "}\n"
        << "}\n";
    ctx.Check(!ec && out.good(),
              "DGSEM machine-readable deterministic flux fixtures written");
  }
}

} // namespace igrlab_tests

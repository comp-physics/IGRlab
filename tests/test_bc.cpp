#include "physics/bc/boundary_state.hpp"

#include <string>

#include "test_utils.hpp"

namespace dg_igr_tests
{

int RunBCTests(TestContext& ctx)
{
  constexpr int dim = 2;
  constexpr double gamma = 1.4;
  const double tol = 1.0e-12;

  mfem::FaceElementTransformations Tr;
  mfem::Vector nor(dim);
  nor = 0.0;
  nor(0) = 1.0;

  // --- Supersonic inlet enforces the prescribed state ---
  const mfem::Vector Uin = EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/2.0,
                                               /*v=*/0.5, /*p=*/1.0, gamma);
  const mfem::Vector Uminus_inlet = EulerConsFromPrim2D(/*rho=*/0.8, /*u=*/-1.0,
                                                        /*v=*/0.25, /*p=*/0.7, gamma);
  mfem::SupersonicInletState inlet(Uin);
  mfem::Vector Uplus(dim + 2);
  inlet.Eval(Uminus_inlet, nor, Tr, Uplus);
  ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));
  for (int i = 0; i < Uplus.Size(); ++i)
  {
    ctx.CheckNear(Uplus(i), Uin(i), tol,
                  "SupersonicInletState enforces Uin component " + std::to_string(i));
  }

  // --- Supersonic outlet is transmissive ---
  mfem::SupersonicOutletState outlet;
  const mfem::Vector Uminus_outlet = EulerConsFromPrim2D(/*rho=*/1.1, /*u=*/0.75,
                                                         /*v=*/-0.25, /*p=*/0.9, gamma);
  outlet.Eval(Uminus_outlet, nor, Tr, Uplus);
  ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));
  for (int i = 0; i < Uplus.Size(); ++i)
  {
    ctx.CheckNear(Uplus(i), Uminus_outlet(i), tol,
                  "SupersonicOutletState copies interior state component " +
                      std::to_string(i));
  }

  // --- Slip wall reflects the normal momentum and preserves tangential ---
  mfem::SlipWallState slip(dim);
  const mfem::Vector Uminus_wall = EulerConsFromPrim2D(/*rho=*/1.25, /*u=*/3.0,
                                                       /*v=*/-1.5, /*p=*/1.0, gamma);
  slip.Eval(Uminus_wall, nor, Tr, Uplus);

  const double rho = Uminus_wall(0);
  const double mx = Uminus_wall(1);
  const double my = Uminus_wall(2);
  const double energy = Uminus_wall(3);

  ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));
  ctx.CheckNear(Uplus(0), rho, tol, "SlipWallState preserves density");
  ctx.CheckNear(Uplus(1), -mx, 1.0e-11, "SlipWallState flips normal momentum");
  ctx.CheckNear(Uplus(2), my, 1.0e-11, "SlipWallState preserves tangential momentum");
  ctx.CheckNear(Uplus(3), energy, 1.0e-11, "SlipWallState preserves energy");

  return ctx.failures;
}

} // namespace dg_igr_tests

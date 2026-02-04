#include "physics/euler/conversions.hpp"
#include "physics/euler/eos.hpp"

#include <cmath>
#include <initializer_list>
#include <string>

#include "test_utils.hpp"

namespace dg_igr_tests
{

int RunEulerTests(TestContext& ctx)
{
  const double gamma = 1.4;
  const double tol = 1e-12;

  // --- Round-trip conversions across dimensions ---
  for (int dim : {1, 2, 3})
  {
    mfem::Vector prim(dim + 2);
    prim(0) = 1.0;  // rho
    for (int d = 0; d < dim; ++d)
    {
      prim(1 + d) = 0.5 * (d + 1);  // velocity components
    }
    prim(1 + dim) = 1.0;  // p

    mfem::Vector cons(dim + 2);
    mfem::EulerPrimToConservative(prim, dim, gamma, cons);

    const double p = mfem::EulerPressureFromConservative(cons, dim, gamma, /*rho_floor=*/0.0);
    ctx.AddParity(p);
    ctx.CheckNear(p, prim(1 + dim), tol,
                  "PressureFromConservative matches primitive pressure (dim=" +
                      std::to_string(dim) + ")");

    mfem::Vector prim_rt(dim + 2);
    mfem::EulerConservativeToPrimitive(cons, dim, gamma, prim_rt);

    for (int i = 0; i < prim.Size(); ++i)
    {
      ctx.CheckNear(prim_rt(i), prim(i), 1e-11,
                    "Primitive round-trip component " + std::to_string(i) +
                        " (dim=" + std::to_string(dim) + ")");
    }
  }

  // --- EOS rho_floor behavior ---
  const int dim = 2;
  mfem::Vector cons_edge(dim + 2);
  cons_edge(0) = 1.0e-14;  // rho (will be floored)
  cons_edge(1) = 1.0e-6;   // rho*u
  cons_edge(2) = 0.0;      // rho*v
  cons_edge(3) = 1.0;      // rho*E

  const double rho_floor = 1.0e-6;
  const double rho_eff = rho_floor;
  const double mom2 = cons_edge(1) * cons_edge(1) + cons_edge(2) * cons_edge(2);
  const double k = 0.5 * mom2 / rho_eff;
  const double expected = (gamma - 1.0) * (cons_edge(3) - k);
  const double p_edge =
      mfem::EulerPressureFromConservative(cons_edge, dim, gamma, rho_floor);
  ctx.AddParity(p_edge);
  ctx.CheckNear(p_edge, expected, 1e-12,
                "EulerPressureFromConservative respects rho_floor in kinetic term");

  // rho_floor should not affect pressure when rho is already above it.
  mfem::Vector prim(dim + 2);
  prim(0) = 1.0;
  prim(1) = 2.0;
  prim(2) = 0.5;
  prim(3) = 1.0;
  mfem::Vector cons(dim + 2);
  mfem::EulerPrimToConservative(prim, dim, gamma, cons);
  const double p_no_floor =
      mfem::EulerPressureFromConservative(cons, dim, gamma, /*rho_floor=*/0.0);
  const double p_with_floor =
      mfem::EulerPressureFromConservative(cons, dim, gamma, /*rho_floor=*/rho_floor);
  ctx.AddParity(p_no_floor + p_with_floor);
  ctx.CheckNear(p_with_floor, p_no_floor, tol,
                "rho_floor does not change pressure when rho is admissible");

  // --- EOS edge-case stress tests ---
  // Near-vacuum density with a floor should remain finite and consistent.
  {
    mfem::Vector cons_nv(dim + 2);
    cons_nv(0) = 1.0e-12;
    cons_nv(1) = 1.0e-8;
    cons_nv(2) = -1.0e-8;
    cons_nv(3) = 1.0e-6;

    const double rho_floor_nv = 1.0e-8;
    const double rho_eff_nv = rho_floor_nv;
    const double mom2_nv = cons_nv(1) * cons_nv(1) + cons_nv(2) * cons_nv(2);
    const double k_nv = 0.5 * mom2_nv / rho_eff_nv;
    const double expected_nv = (gamma - 1.0) * (cons_nv(3) - k_nv);
    const double p_nv =
        mfem::EulerPressureFromConservative(cons_nv, dim, gamma, rho_floor_nv);
    ctx.AddParity(p_nv);

    ctx.Check(std::isfinite(p_nv), "Near-vacuum EOS pressure is finite with rho_floor");
    ctx.CheckNear(p_nv, expected_nv, 1.0e-12,
                  "Near-vacuum EOS pressure matches floored kinetic energy");
  }

  // High kinetic energy should not overflow and should remain finite.
  {
    mfem::Vector cons_hi(dim + 2);
    cons_hi(0) = 1.0;
    cons_hi(1) = 100.0;
    cons_hi(2) = -50.0;
    cons_hi(3) = 7000.0;

    const double p_hi =
        mfem::EulerPressureFromConservative(cons_hi, dim, gamma, /*rho_floor=*/0.0);
    ctx.AddParity(p_hi);
    ctx.Check(std::isfinite(p_hi), "High-kinetic EOS pressure is finite");
    ctx.Check(p_hi > 0.0, "High-kinetic EOS pressure remains positive for admissible energy");
  }

  // Very small/large gamma values should remain algebraically consistent.
  for (double gamma_case : {1.01, 1.4, 5.0})
  {
    mfem::Vector prim_case(dim + 2);
    prim_case(0) = 1.2;
    prim_case(1) = 0.3;
    prim_case(2) = -0.1;
    prim_case(3) = 2.5;

    mfem::Vector cons_case(dim + 2);
    mfem::EulerPrimToConservative(prim_case, dim, gamma_case, cons_case);

    const double p_case = mfem::EulerPressureFromConservative(
        cons_case, dim, gamma_case, /*rho_floor=*/0.0);
    ctx.AddParity(p_case);
    ctx.Check(std::isfinite(p_case),
              "EOS pressure is finite for gamma=" + FormatSci(gamma_case));
    ctx.CheckNear(p_case, prim_case(1 + dim), 1.0e-11,
                  "EOS pressure matches primitive pressure for gamma=" +
                      FormatSci(gamma_case));
  }

  return ctx.failures;
}

} // namespace dg_igr_tests

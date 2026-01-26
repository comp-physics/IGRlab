#include "numerics/limiters/zhang_shu.hpp"

#include "physics/euler/eos.hpp"
#include "limiters_common.hpp"

#include <algorithm>
#include <cmath>

namespace {

static inline double ThetaForPressure(const mfem::Vector &Ubar,
                                      const mfem::Vector &Up,
                                      int dim, double gamma, double p_min,
                                      double theta_max = 1.0, int iters = 12)
{
  mfem::Vector Utest(Ubar.Size());
  Utest = Ubar; Utest.Add(theta_max, Up); Utest.Add(-theta_max, Ubar);
  if (mfem::EulerPressureFromConservative(Utest, dim, gamma, /*rho_floor=*/1e-16) >= p_min)
  { return theta_max; }

  double lo = 0.0, hi = theta_max;
  for (int it = 0; it < iters; ++it)
  {
    const double mid = 0.5*(lo + hi);
    Utest = Ubar; Utest.Add(mid, Up); Utest.Add(-mid, Ubar);
    const double p = mfem::EulerPressureFromConservative(Utest, dim, gamma, /*rho_floor=*/1e-16);
    if (p >= p_min) { lo = mid; } else { hi = mid; }
  }
  return lo;
}

} // namespace

namespace mfem
{

void ApplyZhangShuPositivityLimiter(ParFiniteElementSpace &vfes,
                                    ParGridFunction &U,
                                    int dim, double gamma,
                                    double rho_min, double p_min,
                                    int quad_order, int p_bisect_iters)
{
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Positivity limiter assumes byNODES ordering.");

  const int num_eq = dim + 2;
  const int NE = vfes.GetNE();

  mfem::Array<int> vdofs;
  mfem::Vector elvec;                 // element slice (size = dof*num_eq)
  mfem::DenseMatrix elmat;            // (dof x num_eq) view of element DOFs
  mfem::Vector shape, Up(num_eq), Ubar(num_eq), row(num_eq);

  const int qord = (quad_order > 0) ? quad_order
                                    : (2*vfes.GetFE(0)->GetOrder() + 2);

  for (int i = 0; i < NE; i++)
  {
    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);

    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    // --- compute cell average ---
    dg_igr::limiters_detail::ElementAverage(*fe, *Tr, elmat, num_eq, qord, Ubar);

    // If the average itself is unphysical, collapse element to constant admissible state
    {
      const double rho_bar = std::max(Ubar(0), rho_min);
      double mom2_bar = 0.0;
      for (int d = 0; d < dim; ++d) mom2_bar += Ubar(1 + d) * Ubar(1 + d);
      const double k_bar = 0.5 * mom2_bar / std::max(rho_bar, 1e-16);
      double E_bar = Ubar(1 + dim);
      double p_bar = (gamma - 1.0) * (E_bar - k_bar);

      if (rho_bar <= rho_min || p_bar <= p_min)
      {
        Ubar(0) = rho_bar;
        E_bar = k_bar + p_min/(gamma - 1.0);
        Ubar(1 + dim) = E_bar;

        for (int j = 0; j < dof; ++j) { elmat.GetRow(j, row); row = Ubar; elmat.SetRow(j, row); }
        U.SetSubVector(vdofs, elmat.GetData());
        continue; // done with this element
      }
    }

    // --- positivity points (use same IR as average) ---
    const IntegrationRule &ir = IntRules.Get(Tr->GetGeometryType(), qord);
    shape.SetSize(dof);

    // Step 1: density limiting
    double theta_rho = 1.0;
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, Up);
      const double rho = Up(0);
      if (rho < rho_min)
      {
        const double num = Ubar(0) - rho_min;
        const double den = Ubar(0) - rho;
        if (den > 0.0) { theta_rho = std::min(theta_rho, num / den); }
        else           { theta_rho = 0.0; }
      }
    }
    if (theta_rho < 1.0)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, row);
        dg_igr::limiters_detail::BlendRow(row, Ubar, theta_rho);
        elmat.SetRow(j, row);
      }
    }

    // Step 2: pressure limiting
    double theta_p = 1.0;
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, Up); // current point state (after rho limit)
      const double p = EulerPressureFromConservative(Up, dim, gamma, /*rho_floor=*/1e-16);
      if (p < p_min)
      {
        theta_p = std::min(theta_p,
                           ThetaForPressure(Ubar, Up, dim, gamma, p_min,
                                            /*theta_max=*/theta_rho,
                                            p_bisect_iters));
      }
    }
    if (theta_p < 1.0)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, row);
        dg_igr::limiters_detail::BlendRow(row, Ubar, theta_p);
        elmat.SetRow(j, row);
      }
    }

    U.SetSubVector(vdofs, elmat.GetData());
  }
}

} // namespace mfem

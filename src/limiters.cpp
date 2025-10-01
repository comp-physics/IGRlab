#include "mfem.hpp"
#include "limiters.hpp"

// Blend helper: U := Ubar + theta * (U - Ubar)
static inline void BlendRow(mfem::Vector &row, const mfem::Vector &ubar, double theta)
{
  for (int k = 0; k < row.Size(); ++k) { row(k) = ubar(k) + theta * (row(k) - ubar(k)); }
}

// Evaluate pressure from conservative state
static inline double Pressure(const mfem::Vector &U, int dim, double gamma)
{
  const double rho = U(0);
  const double E   = U(1 + dim);
  double mom2 = 0.0;
  for (int d = 0; d < dim; ++d) mom2 += U(1 + d) * U(1 + d);
  const double k = 0.5 * mom2 / rho;
  return (gamma - 1.0) * (E - k);
}

// Find the smallest theta in [0,theta_max] such that p(Ubar + theta*(Up - Ubar)) >= p_min
// Use safe bisection because p(theta) is concave in admissible set.
static double ThetaForPressure(const mfem::Vector &Ubar,
                               const mfem::Vector &Up,
                               int dim, double gamma, double p_min,
                               double theta_max = 1.0, int iters = 12)
{
  // Quick accept
  mfem::Vector Utest(Ubar.Size());
  Utest = Ubar; Utest.Add(theta_max, Up); Utest.Add(-theta_max, Ubar); // Ubar + theta_max*(Up-Ubar)
  if (Pressure(Utest, dim, gamma) >= p_min) { return theta_max; }

  double lo = 0.0, hi = theta_max;
  for (int it = 0; it < iters; ++it)
  {
    const double mid = 0.5*(lo + hi);
    Utest = Ubar; Utest.Add(mid, Up); Utest.Add(-mid, Ubar);
    const double p = Pressure(Utest, dim, gamma);
    if (p >= p_min) { lo = mid; } else { hi = mid; }
  }
  return lo;
}

// Compute cell average of U on element i by quadrature of order 'qord'.
static void ElementAverage(const mfem::FiniteElement &fe,
                           mfem::ElementTransformation &Tr,
                           const mfem::DenseMatrix &elfun_mat,
                           int num_eq, int qord,
                           mfem::Vector &Ubar_out)
{
  const mfem::IntegrationRule &ir = mfem::IntRules.Get(Tr.GetGeometryType(), qord);
  mfem::Vector shape(fe.GetDof()), Upt(num_eq);
  Ubar_out.SetSize(num_eq); Ubar_out = 0.0;

  double vol = 0.0;
  for (int i = 0; i < ir.GetNPoints(); i++)
  {
    const mfem::IntegrationPoint &ip = ir.IntPoint(i);
    Tr.SetIntPoint(&ip);
    fe.CalcShape(ip, shape);
    elfun_mat.MultTranspose(shape, Upt);
    const double w = ip.weight * Tr.Weight();
    Ubar_out.Add(w, Upt);
    vol += w;
  }
  Ubar_out /= vol;
}

void ApplyZhangShuPositivityLimiter(mfem::ParFiniteElementSpace &vfes,
                                    mfem::ParGridFunction &U,
                                    int dim, double gamma,
                                    double rho_min, double p_min,
                                    int quad_order, int p_bisect_iters)
{
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "This implementation assumes byNODES ordering (your code already uses it).");

  const int num_eq = dim + 2;
  const int NE = vfes.GetNE();

  mfem::Array<int> vdofs;
  mfem::Vector elvec;                 // element slice (size = dof*num_eq)
  mfem::DenseMatrix elmat;            // (dof × num_eq) view of element DOFs
  mfem::Vector shape, Up(num_eq), Ubar(num_eq), row(num_eq);

  for (int i = 0; i < NE; i++)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);

    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    // --- compute cell average ---
    ElementAverage(*fe, *Tr, elmat, num_eq, quad_order, Ubar);

    // If the average itself is unphysical, collapse element to an admissible constant
    // state using average rho and momentum; lift energy to make p = p_min.
    {
      const double rho_bar = std::max(Ubar(0), rho_min);
      double mom2_bar = 0.0;
      for (int d = 0; d < dim; ++d) mom2_bar += Ubar(1 + d) * Ubar(1 + d);
      const double k_bar = 0.5 * mom2_bar / rho_bar;
      double E_bar = Ubar(1 + dim);
      double p_bar = (gamma - 1.0) * (E_bar - k_bar);

      if (rho_bar <= rho_min || p_bar <= p_min)
      {
        Ubar(0) = rho_bar;
        for (int d = 0; d < dim; ++d) { /* leave momentum average as is */ }
        E_bar = k_bar + p_min/(gamma - 1.0);
        Ubar(1 + dim) = E_bar;

        for (int j = 0; j < dof; ++j) { elmat.GetRow(j, row); row = Ubar; elmat.SetRow(j, row); }
        U.SetSubVector(vdofs, elmat.GetData());
        continue; // done with this element
      }
    }

    // --- choose positivity points (same IR used for average is fine) ---
    const mfem::IntegrationRule &ir = mfem::IntRules.Get(Tr->GetGeometryType(), quad_order);
    shape.SetSize(dof);

    // Step 1: density limiting (compute theta_rho ∈ (0,1])
    double theta_rho = 1.0;
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
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

    // Apply density theta to all nodal rows (affine blend per conservative component)
    if (theta_rho < 1.0)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, row);
        BlendRow(row, Ubar, theta_rho);
        elmat.SetRow(j, row);
      }
    }

    // Step 2: pressure limiting (compute theta_p ≤ theta_rho)
    double theta_p = 1.0;
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, Up);                  // Up is *current* point state (after rho limit)
      const double p = Pressure(Up, dim, gamma);
      if (p < p_min)
      {
        // Need a theta ≤ current 1.0 that raises p to p_min at this point
        // Build the *original* point state before blending with theta (relative to Ubar).
        // But since we only need monotone path from Ubar to Up, use Up directly:
        // find theta in [0,1] such that p(Ubar + theta*(Up - Ubar)) >= p_min
        theta_p = std::min(theta_p, ThetaForPressure(Ubar, Up, dim, gamma, p_min,
                                                     /*theta_max=*/theta_rho, p_bisect_iters));
      }
    }

    if (theta_p < 1.0)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, row);
        BlendRow(row, Ubar, theta_p);
        elmat.SetRow(j, row);
      }
    }

    // scatter back
    U.SetSubVector(vdofs, elmat.GetData());
  }
}

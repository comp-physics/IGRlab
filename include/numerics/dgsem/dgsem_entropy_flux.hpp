#pragma once

/**
 * @file numerics/dgsem/dgsem_entropy_flux.hpp
 * @brief Entropy-conservative flux helpers for DGSEM (Euler).
 * @ingroup numerics_dgsem
 */

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mfem
{
namespace dgsem
{

/// @brief Entropy-conservative two-point flux choices for Euler.
enum class EntropyFluxType
{
  Chandrashekar,
  Ranocha
};

/// @brief Entropy Jacobian evaluation mode for LF entropy-variable dissipation.
enum class EntropyJacobianMode
{
  FiniteDifference,
  Analytic
};

inline double LogMean(double a, double b)
{
  MFEM_VERIFY(a > 0.0 && b > 0.0, "LogMean requires positive inputs.");
  const double diff = b - a;
  const double sum = a + b;
  const double tol = 1.0e-12 * std::max(1.0, std::abs(sum));
  if (std::abs(diff) < tol)
  {
    return 0.5 * sum;
  }
  return diff / (std::log(b) - std::log(a));
}

inline double InvLogMean(double a, double b)
{
  MFEM_VERIFY(a > 0.0 && b > 0.0, "InvLogMean requires positive inputs.");
  const double diff = b - a;
  const double sum = a + b;
  const double tol = 1.0e-12 * std::max(1.0, std::abs(sum));
  if (std::abs(diff) < tol)
  {
    return 2.0 / sum;
  }
  return (std::log(b) - std::log(a)) / diff;
}

inline void EulerPrimitive(const Vector& u,
                           int dim,
                           double gamma,
                           double sigma,
                           double& rho,
                           double& v1,
                           double& v2,
                           double& p,
                           double& rho_e)
{
  MFEM_VERIFY(dim == 1 || dim == 2,
              "EulerPrimitive supports dim=1 or dim=2.");
  const int num_eq = dim + 2;
  MFEM_VERIFY(u.Size() == num_eq, "EulerPrimitive: unexpected state size.");

  rho = u(0);
  rho_e = u(num_eq - 1);
  v1 = u(1) / rho;
  v2 = (dim == 2) ? u(2) / rho : 0.0;
  const double v_sq = v1 * v1 + v2 * v2;
  p = (gamma - 1.0) * (rho_e - 0.5 * rho * v_sq) + sigma;
}

inline void EulerEntropyVariables(const Vector& u,
                                  int dim,
                                  double gamma,
                                  double sigma,
                                  Vector& w)
{
  double rho, v1, v2, p, rho_e;
  EulerPrimitive(u, dim, gamma, sigma, rho, v1, v2, p, rho_e);
  MFEM_VERIFY(p > 0.0, "EulerEntropyVariables: non-positive pressure.");
  const double v_sq = v1 * v1 + v2 * v2;
  const double s = std::log(p) - gamma * std::log(rho);
  const double rho_p = rho / p;

  w.SetSize(dim + 2);
  w(0) = (gamma - s) / (gamma - 1.0) - 0.5 * rho_p * v_sq;
  w(1) = rho_p * v1;
  if (dim == 2)
  {
    w(2) = rho_p * v2;
    w(3) = -rho_p;
  }
  else
  {
    w(2) = -rho_p;
  }
}

inline bool EulerStateIsValid(const Vector& u,
                              int dim,
                              double gamma,
                              double sigma)
{
  double rho, v1, v2, p, rho_e;
  EulerPrimitive(u, dim, gamma, sigma, rho, v1, v2, p, rho_e);
  return (rho > 0.0) && (p > 0.0);
}

inline void BuildMeanState(const Vector& ul,
                           const Vector& ur,
                           int dim,
                           double gamma,
                           double sigma_l,
                           double sigma_r,
                           Vector& u_bar,
                           double& sigma_bar)
{
  double rho_l, v1_l, v2_l, p_l, rhoe_l;
  double rho_r, v1_r, v2_r, p_r, rhoe_r;
  EulerPrimitive(ul, dim, gamma, sigma_l, rho_l, v1_l, v2_l, p_l, rhoe_l);
  EulerPrimitive(ur, dim, gamma, sigma_r, rho_r, v1_r, v2_r, p_r, rhoe_r);

  const double rho_bar = LogMean(rho_l, rho_r);
  const double p_bar = LogMean(p_l, p_r);
  const double v1_bar = 0.5 * (v1_l + v1_r);
  const double v2_bar = (dim == 2) ? 0.5 * (v2_l + v2_r) : 0.0;
  sigma_bar = 0.5 * (sigma_l + sigma_r);

  u_bar.SetSize(dim + 2);
  u_bar(0) = rho_bar;
  u_bar(1) = rho_bar * v1_bar;
  if (dim == 2)
  {
    u_bar(2) = rho_bar * v2_bar;
  }
  const double v_sq = v1_bar * v1_bar + v2_bar * v2_bar;
  const double p_eff = p_bar;
  const double rho_e = (p_eff - sigma_bar) / (gamma - 1.0) + 0.5 * rho_bar * v_sq;
  u_bar(dim + 1) = rho_e;
}

inline void EntropyJacobian(const Vector& u,
                            int dim,
                            double gamma,
                            double sigma,
                            DenseMatrix& J)
{
  const int n = u.Size();
  J.SetSize(n);
  J = 0.0;

  Vector u_pert(n);
  Vector w_plus(n);
  Vector w_minus(n);

  for (int i = 0; i < n; ++i)
  {
    double eps = 1.0e-7 * std::max(1.0, std::abs(u(i)));
    bool ok = false;
    for (int attempt = 0; attempt < 12; ++attempt)
    {
      u_pert = u;
      u_pert(i) += eps;
      if (!EulerStateIsValid(u_pert, dim, gamma, sigma))
      {
        eps *= 0.5;
        continue;
      }
      EulerEntropyVariables(u_pert, dim, gamma, sigma, w_plus);

      u_pert = u;
      u_pert(i) -= eps;
      if (!EulerStateIsValid(u_pert, dim, gamma, sigma))
      {
        eps *= 0.5;
        continue;
      }
      EulerEntropyVariables(u_pert, dim, gamma, sigma, w_minus);

      ok = true;
      break;
    }

    MFEM_VERIFY(ok, "EntropyJacobian finite-difference failed.");

    for (int r = 0; r < n; ++r)
    {
      J(r, i) = (w_plus(r) - w_minus(r)) / (2.0 * eps);
    }
  }
}

inline void EntropyJacobianAnalytic(const Vector& u,
                                    int dim,
                                    double gamma,
                                    double sigma,
                                    DenseMatrix& J)
{
  MFEM_VERIFY(dim == 1 || dim == 2,
              "EntropyJacobianAnalytic supports dim=1 or dim=2.");
  const int n = u.Size();
  J.SetSize(n);
  J = 0.0;

  const double rho = u(0);
  MFEM_VERIFY(rho > 0.0, "EntropyJacobianAnalytic requires rho > 0.");
  const double inv_rho = 1.0 / rho;
  const double gm1 = gamma - 1.0;

  if (dim == 1)
  {
    const double m = u(1);
    const double E = u(2);
    const double v = m * inv_rho;
    const double q2 = v * v;
    const double p = gm1 * (E - 0.5 * rho * q2) + sigma;
    MFEM_VERIFY(p > 0.0, "EntropyJacobianAnalytic requires p > 0.");
    const double inv_p = 1.0 / p;
    const double inv_p2 = inv_p * inv_p;

    const double dp_drho = 0.5 * gm1 * q2;
    const double dp_dm = -gm1 * v;
    const double dp_dE = gm1;

    const double dq2_drho = -2.0 * q2 * inv_rho;
    const double dq2_dm = 2.0 * v * inv_rho;

    const double dv_drho = -v * inv_rho;
    const double dv_dm = inv_rho;

    const double beta = 0.5 * rho * inv_p;
    const double dbeta_drho = 0.5 * (inv_p - rho * inv_p2 * dp_drho);
    const double dbeta_dm = -0.5 * rho * inv_p2 * dp_dm;
    const double dbeta_dE = -0.5 * rho * inv_p2 * dp_dE;

    const double ds_drho = inv_p * dp_drho - gamma * inv_rho;
    const double ds_dm = inv_p * dp_dm;
    const double ds_dE = inv_p * dp_dE;

    // w1
    J(0, 0) = -(1.0 / gm1) * ds_drho - dbeta_drho * q2 - beta * dq2_drho;
    J(0, 1) = -(1.0 / gm1) * ds_dm - dbeta_dm * q2 - beta * dq2_dm;
    J(0, 2) = -(1.0 / gm1) * ds_dE - dbeta_dE * q2;

    // w2 = 2 beta v
    J(1, 0) = 2.0 * dbeta_drho * v + 2.0 * beta * dv_drho;
    J(1, 1) = 2.0 * dbeta_dm * v + 2.0 * beta * dv_dm;
    J(1, 2) = 2.0 * dbeta_dE * v;

    // w3 = -2 beta
    J(2, 0) = -2.0 * dbeta_drho;
    J(2, 1) = -2.0 * dbeta_dm;
    J(2, 2) = -2.0 * dbeta_dE;

    return;
  }

  const double m1 = u(1);
  const double m2 = u(2);
  const double E = u(3);
  const double v1 = m1 * inv_rho;
  const double v2 = m2 * inv_rho;
  const double q2 = v1 * v1 + v2 * v2;
  const double p = gm1 * (E - 0.5 * rho * q2) + sigma;
  MFEM_VERIFY(p > 0.0, "EntropyJacobianAnalytic requires p > 0.");
  const double inv_p = 1.0 / p;
  const double inv_p2 = inv_p * inv_p;

  const double dp_drho = 0.5 * gm1 * q2;
  const double dp_dm1 = -gm1 * v1;
  const double dp_dm2 = -gm1 * v2;
  const double dp_dE = gm1;

  const double dq2_drho = -2.0 * q2 * inv_rho;
  const double dq2_dm1 = 2.0 * v1 * inv_rho;
  const double dq2_dm2 = 2.0 * v2 * inv_rho;

  const double dv1_drho = -v1 * inv_rho;
  const double dv1_dm1 = inv_rho;
  const double dv2_drho = -v2 * inv_rho;
  const double dv2_dm2 = inv_rho;

  const double beta = 0.5 * rho * inv_p;
  const double dbeta_drho = 0.5 * (inv_p - rho * inv_p2 * dp_drho);
  const double dbeta_dm1 = -0.5 * rho * inv_p2 * dp_dm1;
  const double dbeta_dm2 = -0.5 * rho * inv_p2 * dp_dm2;
  const double dbeta_dE = -0.5 * rho * inv_p2 * dp_dE;

  const double ds_drho = inv_p * dp_drho - gamma * inv_rho;
  const double ds_dm1 = inv_p * dp_dm1;
  const double ds_dm2 = inv_p * dp_dm2;
  const double ds_dE = inv_p * dp_dE;

  // w1
  J(0, 0) = -(1.0 / gm1) * ds_drho - dbeta_drho * q2 - beta * dq2_drho;
  J(0, 1) = -(1.0 / gm1) * ds_dm1 - dbeta_dm1 * q2 - beta * dq2_dm1;
  J(0, 2) = -(1.0 / gm1) * ds_dm2 - dbeta_dm2 * q2 - beta * dq2_dm2;
  J(0, 3) = -(1.0 / gm1) * ds_dE - dbeta_dE * q2;

  // w2 = 2 beta v1
  J(1, 0) = 2.0 * dbeta_drho * v1 + 2.0 * beta * dv1_drho;
  J(1, 1) = 2.0 * dbeta_dm1 * v1 + 2.0 * beta * dv1_dm1;
  J(1, 2) = 2.0 * dbeta_dm2 * v1;
  J(1, 3) = 2.0 * dbeta_dE * v1;

  // w3 = 2 beta v2
  J(2, 0) = 2.0 * dbeta_drho * v2 + 2.0 * beta * dv2_drho;
  J(2, 1) = 2.0 * dbeta_dm1 * v2;
  J(2, 2) = 2.0 * dbeta_dm2 * v2 + 2.0 * beta * dv2_dm2;
  J(2, 3) = 2.0 * dbeta_dE * v2;

  // w4 = -2 beta
  J(3, 0) = -2.0 * dbeta_drho;
  J(3, 1) = -2.0 * dbeta_dm1;
  J(3, 2) = -2.0 * dbeta_dm2;
  J(3, 3) = -2.0 * dbeta_dE;
}

inline void ChandrashekarFlux1D(const Vector& ul,
                                const Vector& ur,
                                double gamma,
                                double sigma_l,
                                double sigma_r,
                                Vector& flux)
{
  double rho_l, v_l, v2_l, p_l, rhoe_l;
  double rho_r, v_r, v2_r, p_r, rhoe_r;
  EulerPrimitive(ul, /*dim=*/1, gamma, sigma_l, rho_l, v_l, v2_l, p_l, rhoe_l);
  EulerPrimitive(ur, /*dim=*/1, gamma, sigma_r, rho_r, v_r, v2_r, p_r, rhoe_r);

  const double beta_l = 0.5 * rho_l / p_l;
  const double beta_r = 0.5 * rho_r / p_r;
  const double rho_avg = 0.5 * (rho_l + rho_r);
  const double rho_mean = LogMean(rho_l, rho_r);
  const double beta_mean = LogMean(beta_l, beta_r);
  const double beta_avg = 0.5 * (beta_l + beta_r);
  const double v_avg = 0.5 * (v_l + v_r);
  const double p_mean = 0.5 * rho_avg / beta_avg;
  const double v_sq_avg = 0.5 * (v_l * v_l + v_r * v_r);

  flux.SetSize(3);
  const double f1 = rho_mean * v_avg;
  const double f2 = f1 * v_avg + p_mean;
  const double f3 =
      f1 * 0.5 * (1.0 / ((gamma - 1.0) * beta_mean) - v_sq_avg) + f2 * v_avg;
  flux(0) = f1;
  flux(1) = f2;
  flux(2) = f3;
}

inline void ChandrashekarFlux2D(const Vector& ul,
                                const Vector& ur,
                                int dir,
                                double gamma,
                                double sigma_l,
                                double sigma_r,
                                Vector& flux)
{
  double rho_l, v1_l, v2_l, p_l, rhoe_l;
  double rho_r, v1_r, v2_r, p_r, rhoe_r;
  EulerPrimitive(ul, /*dim=*/2, gamma, sigma_l, rho_l, v1_l, v2_l, p_l, rhoe_l);
  EulerPrimitive(ur, /*dim=*/2, gamma, sigma_r, rho_r, v1_r, v2_r, p_r, rhoe_r);

  const double beta_l = 0.5 * rho_l / p_l;
  const double beta_r = 0.5 * rho_r / p_r;
  const double rho_avg = 0.5 * (rho_l + rho_r);
  const double rho_mean = LogMean(rho_l, rho_r);
  const double beta_mean = LogMean(beta_l, beta_r);
  const double beta_avg = 0.5 * (beta_l + beta_r);
  const double v1_avg = 0.5 * (v1_l + v1_r);
  const double v2_avg = 0.5 * (v2_l + v2_r);
  const double p_mean = 0.5 * rho_avg / beta_avg;
  const double kin_l = 0.5 * (v1_l * v1_l + v2_l * v2_l);
  const double kin_r = 0.5 * (v1_r * v1_r + v2_r * v2_r);
  const double v_sq_avg = kin_l + kin_r;

  flux.SetSize(4);
  if (dir == 0)
  {
    const double f1 = rho_mean * v1_avg;
    const double f2 = f1 * v1_avg + p_mean;
    const double f3 = f1 * v2_avg;
    const double f4 =
        f1 * 0.5 * (1.0 / ((gamma - 1.0) * beta_mean) - v_sq_avg) +
        f2 * v1_avg + f3 * v2_avg;
    flux(0) = f1;
    flux(1) = f2;
    flux(2) = f3;
    flux(3) = f4;
  }
  else
  {
    const double f1 = rho_mean * v2_avg;
    const double f2 = f1 * v1_avg;
    const double f3 = f1 * v2_avg + p_mean;
    const double f4 =
        f1 * 0.5 * (1.0 / ((gamma - 1.0) * beta_mean) - v_sq_avg) +
        f2 * v1_avg + f3 * v2_avg;
    flux(0) = f1;
    flux(1) = f2;
    flux(2) = f3;
    flux(3) = f4;
  }
}

inline void RanochaFlux1D(const Vector& ul,
                          const Vector& ur,
                          double gamma,
                          double sigma_l,
                          double sigma_r,
                          Vector& flux)
{
  double rho_l, v_l, v2_l, p_l, rhoe_l;
  double rho_r, v_r, v2_r, p_r, rhoe_r;
  EulerPrimitive(ul, /*dim=*/1, gamma, sigma_l, rho_l, v_l, v2_l, p_l, rhoe_l);
  EulerPrimitive(ur, /*dim=*/1, gamma, sigma_r, rho_r, v_r, v2_r, p_r, rhoe_r);

  const double rho_mean = LogMean(rho_l, rho_r);
  const double inv_rho_p_mean =
      p_l * p_r * InvLogMean(rho_l * p_r, rho_r * p_l);
  const double v_avg = 0.5 * (v_l + v_r);
  const double p_avg = 0.5 * (p_l + p_r);
  const double v_sq_avg = 0.5 * (v_l * v_r);

  flux.SetSize(3);
  const double f1 = rho_mean * v_avg;
  const double f2 = f1 * v_avg + p_avg;
  const double f3 =
      f1 * (v_sq_avg + inv_rho_p_mean / (gamma - 1.0)) +
      0.5 * (p_l * v_r + p_r * v_l);
  flux(0) = f1;
  flux(1) = f2;
  flux(2) = f3;
}

inline void RanochaFlux2D(const Vector& ul,
                          const Vector& ur,
                          int dir,
                          double gamma,
                          double sigma_l,
                          double sigma_r,
                          Vector& flux)
{
  double rho_l, v1_l, v2_l, p_l, rhoe_l;
  double rho_r, v1_r, v2_r, p_r, rhoe_r;
  EulerPrimitive(ul, /*dim=*/2, gamma, sigma_l, rho_l, v1_l, v2_l, p_l, rhoe_l);
  EulerPrimitive(ur, /*dim=*/2, gamma, sigma_r, rho_r, v1_r, v2_r, p_r, rhoe_r);

  const double rho_mean = LogMean(rho_l, rho_r);
  const double inv_rho_p_mean =
      p_l * p_r * InvLogMean(rho_l * p_r, rho_r * p_l);
  const double v1_avg = 0.5 * (v1_l + v1_r);
  const double v2_avg = 0.5 * (v2_l + v2_r);
  const double p_avg = 0.5 * (p_l + p_r);
  const double v_sq_avg = 0.5 * (v1_l * v1_r + v2_l * v2_r);

  flux.SetSize(4);
  if (dir == 0)
  {
    const double f1 = rho_mean * v1_avg;
    const double f2 = f1 * v1_avg + p_avg;
    const double f3 = f1 * v2_avg;
    const double f4 =
        f1 * (v_sq_avg + inv_rho_p_mean / (gamma - 1.0)) +
        0.5 * (p_l * v1_r + p_r * v1_l);
    flux(0) = f1;
    flux(1) = f2;
    flux(2) = f3;
    flux(3) = f4;
  }
  else
  {
    const double f1 = rho_mean * v2_avg;
    const double f2 = f1 * v1_avg;
    const double f3 = f1 * v2_avg + p_avg;
    const double f4 =
        f1 * (v_sq_avg + inv_rho_p_mean / (gamma - 1.0)) +
        0.5 * (p_l * v2_r + p_r * v2_l);
    flux(0) = f1;
    flux(1) = f2;
    flux(2) = f3;
    flux(3) = f4;
  }
}

inline void EntropyConservativeFlux(const Vector& ul,
                                    const Vector& ur,
                                    int dir,
                                    double gamma,
                                    EntropyFluxType type,
                                    Vector& flux,
                                    double sigma_l = 0.0,
                                    double sigma_r = 0.0)
{
  const int num_eq = ul.Size();
  MFEM_VERIFY(num_eq == ur.Size(),
              "EntropyConservativeFlux requires matching state sizes.");
  if (num_eq == 3)
  {
    if (type == EntropyFluxType::Chandrashekar)
    {
      ChandrashekarFlux1D(ul, ur, gamma, sigma_l, sigma_r, flux);
    }
    else
    {
      RanochaFlux1D(ul, ur, gamma, sigma_l, sigma_r, flux);
    }
    return;
  }
  if (num_eq == 4)
  {
    if (type == EntropyFluxType::Chandrashekar)
    {
      ChandrashekarFlux2D(ul, ur, dir, gamma, sigma_l, sigma_r, flux);
    }
    else
    {
      RanochaFlux2D(ul, ur, dir, gamma, sigma_l, sigma_r, flux);
    }
    return;
  }
  MFEM_VERIFY(false, "EntropyConservativeFlux supports Euler in 1D or 2D.");
}

inline void EntropyConservativeFluxDotN(const Vector& ul,
                                        const Vector& ur,
                                        const Vector& normal,
                                        double gamma,
                                        EntropyFluxType type,
                                        Vector& flux,
                                        double sigma_l = 0.0,
                                        double sigma_r = 0.0)
{
  const int num_eq = ul.Size();
  const int dim = normal.Size();
  MFEM_VERIFY(num_eq == ur.Size(),
              "EntropyConservativeFluxDotN requires matching state sizes.");
  MFEM_VERIFY(dim == num_eq - 2,
              "EntropyConservativeFluxDotN requires normal size = dim.");
  flux.SetSize(num_eq);
  flux = 0.0;

  Vector f_dir(num_eq);
  for (int d = 0; d < dim; ++d)
  {
    EntropyConservativeFlux(ul, ur, d, gamma, type, f_dir, sigma_l, sigma_r);
    for (int eq = 0; eq < num_eq; ++eq)
    {
      flux(eq) += normal(d) * f_dir(eq);
    }
  }
}

inline void WintersDissipation1D(const Vector& ul,
                                 const Vector& ur,
                                 const Vector& normal,
                                 double gamma,
                                 double sigma_l,
                                 double sigma_r,
                                 Vector& diss)
{
  const double norm = normal.Norml2();
  if (norm == 0.0)
  {
    diss.SetSize(ul.Size());
    diss = 0.0;
    return;
  }
  const double n1 = normal(0) / norm;

  double rho_l, v_l, v2_l, p_l, rhoe_l;
  double rho_r, v_r, v2_r, p_r, rhoe_r;
  EulerPrimitive(ul, /*dim=*/1, gamma, sigma_l, rho_l, v_l, v2_l, p_l, rhoe_l);
  EulerPrimitive(ur, /*dim=*/1, gamma, sigma_r, rho_r, v_r, v2_r, p_r, rhoe_r);

  const double b_l = rho_l / (2.0 * p_l);
  const double b_r = rho_r / (2.0 * p_r);
  const double rho_log = LogMean(rho_l, rho_r);
  const double b_log = LogMean(b_l, b_r);
  const double v_avg = 0.5 * (v_l + v_r);
  const double p_avg = 0.5 * (rho_l + rho_r) / (b_l + b_r);
  const double v_sq_bar = v_l * v_r;
  const double h_bar = gamma / (2.0 * b_log * (gamma - 1.0)) + 0.5 * v_sq_bar;
  const double c_bar = std::sqrt(gamma * p_avg / rho_log);
  const double v_avg_normal = v_avg * n1;
  const double v_minus_c = v_avg - c_bar * n1;
  const double v_plus_c = v_avg + c_bar * n1;

  Vector w_l(3), w_r(3);
  EulerEntropyVariables(ul, /*dim=*/1, gamma, sigma_l, w_l);
  EulerEntropyVariables(ur, /*dim=*/1, gamma, sigma_r, w_r);
  Vector w_jump(3);
  w_jump = w_r;
  w_jump -= w_l;

  const double lambda1 = std::abs(v_avg_normal - c_bar) * rho_log / (2.0 * gamma);
  const double lambda2 = std::abs(v_avg_normal) * rho_log * (gamma - 1.0) / gamma;
  const double lambda3 = std::abs(v_avg_normal + c_bar) * rho_log / (2.0 * gamma);
  const double lambda4 = std::abs(v_avg_normal) * p_avg;

  const double w1_jump = w_jump(0);
  const double w2_jump = w_jump(1);
  const double w3_jump = w_jump(2);

  const double w1 = lambda1 * (w1_jump + v_minus_c * w2_jump +
                               (h_bar - c_bar * v_avg_normal) * w3_jump);
  const double w2 = lambda2 * (w1_jump + v_avg * w2_jump +
                               0.5 * v_sq_bar * w3_jump);
  const double w3 = lambda3 * (w1_jump + v_plus_c * w2_jump +
                               (h_bar + c_bar * v_avg_normal) * w3_jump);

  const double diss_rho = w1 + w2 + w3;
  const double diss_rho_v = w1 * v_minus_c + w2 * v_avg + w3 * v_plus_c;
  const double diss_rhoe =
      w1 * (h_bar - c_bar * v_avg_normal) +
      w2 * 0.5 * v_sq_bar +
      w3 * (h_bar + c_bar * v_avg_normal) +
      lambda4 * (w3_jump * (v_avg * v_avg - v_avg_normal * v_avg_normal));

  diss.SetSize(3);
  diss(0) = -0.5 * diss_rho * norm;
  diss(1) = -0.5 * diss_rho_v * norm;
  diss(2) = -0.5 * diss_rhoe * norm;
}

inline void WintersDissipation2D(const Vector& ul,
                                 const Vector& ur,
                                 const Vector& normal,
                                 double gamma,
                                 double sigma_l,
                                 double sigma_r,
                                 Vector& diss)
{
  const double norm = normal.Norml2();
  if (norm == 0.0)
  {
    diss.SetSize(ul.Size());
    diss = 0.0;
    return;
  }
  const double nx = normal(0) / norm;
  const double ny = normal(1) / norm;

  double rho_l, v1_l, v2_l, p_l, rhoe_l;
  double rho_r, v1_r, v2_r, p_r, rhoe_r;
  EulerPrimitive(ul, /*dim=*/2, gamma, sigma_l, rho_l, v1_l, v2_l, p_l, rhoe_l);
  EulerPrimitive(ur, /*dim=*/2, gamma, sigma_r, rho_r, v1_r, v2_r, p_r, rhoe_r);

  const double b_l = rho_l / (2.0 * p_l);
  const double b_r = rho_r / (2.0 * p_r);
  const double rho_log = LogMean(rho_l, rho_r);
  const double b_log = LogMean(b_l, b_r);
  const double v1_avg = 0.5 * (v1_l + v1_r);
  const double v2_avg = 0.5 * (v2_l + v2_r);
  const double p_avg = 0.5 * (rho_l + rho_r) / (b_l + b_r);
  const double v_sq_bar = v1_l * v1_r + v2_l * v2_r;
  const double h_bar = gamma / (2.0 * b_log * (gamma - 1.0)) + 0.5 * v_sq_bar;
  const double c_bar = std::sqrt(gamma * p_avg / rho_log);
  const double v_avg_normal = v1_avg * nx + v2_avg * ny;

  const double lambda1 = std::abs(v_avg_normal - c_bar) * rho_log / (2.0 * gamma);
  const double lambda2 = std::abs(v_avg_normal) * rho_log * (gamma - 1.0) / gamma;
  const double lambda3 = std::abs(v_avg_normal + c_bar) * rho_log / (2.0 * gamma);
  const double lambda4 = std::abs(v_avg_normal) * p_avg;

  const double v1_minus_c = v1_avg - c_bar * nx;
  const double v2_minus_c = v2_avg - c_bar * ny;
  const double v1_plus_c = v1_avg + c_bar * nx;
  const double v2_plus_c = v2_avg + c_bar * ny;
  const double v1_tan = v1_avg - v_avg_normal * nx;
  const double v2_tan = v2_avg - v_avg_normal * ny;

  Vector w_l(4), w_r(4);
  EulerEntropyVariables(ul, /*dim=*/2, gamma, sigma_l, w_l);
  EulerEntropyVariables(ur, /*dim=*/2, gamma, sigma_r, w_r);
  Vector w_jump(4);
  w_jump = w_r;
  w_jump -= w_l;

  const double w1_jump = w_jump(0);
  const double w2_jump = w_jump(1);
  const double w3_jump = w_jump(2);
  const double w4_jump = w_jump(3);

  const double v_minus_dot = v1_minus_c * w2_jump + v2_minus_c * w3_jump;
  const double v_plus_dot = v1_plus_c * w2_jump + v2_plus_c * w3_jump;
  const double v_avg_dot = v1_avg * w2_jump + v2_avg * w3_jump;

  const double w1 = lambda1 * (w1_jump + v_minus_dot +
                               (h_bar - c_bar * v_avg_normal) * w4_jump);
  const double w2 = lambda2 * (w1_jump + v_avg_dot +
                               0.5 * v_sq_bar * w4_jump);
  const double w3 = lambda3 * (w1_jump + v_plus_dot +
                               (h_bar + c_bar * v_avg_normal) * w4_jump);

  const double w_normal_jump = w2_jump * nx + w3_jump * ny;
  const double diss_rho = w1 + w2 + w3;
  const double diss_rho_v1 =
      w1 * v1_minus_c + w2 * v1_avg + w3 * v1_plus_c +
      lambda4 * (w2_jump - nx * w_normal_jump + w4_jump * v1_tan);
  const double diss_rho_v2 =
      w1 * v2_minus_c + w2 * v2_avg + w3 * v2_plus_c +
      lambda4 * (w3_jump - ny * w_normal_jump + w4_jump * v2_tan);

  const double v_tan_dot = v1_tan * w2_jump + v2_tan * w3_jump;
  const double v_avg_sq = v1_avg * v1_avg + v2_avg * v2_avg;
  const double diss_rhoe =
      w1 * (h_bar - c_bar * v_avg_normal) +
      w2 * 0.5 * v_sq_bar +
      w3 * (h_bar + c_bar * v_avg_normal) +
      lambda4 * (v_tan_dot + w4_jump * (v_avg_sq - v_avg_normal * v_avg_normal));

  diss.SetSize(4);
  diss(0) = -0.5 * diss_rho * norm;
  diss(1) = -0.5 * diss_rho_v1 * norm;
  diss(2) = -0.5 * diss_rho_v2 * norm;
  diss(3) = -0.5 * diss_rhoe * norm;
}

inline void WintersDissipation(const Vector& ul,
                               const Vector& ur,
                               const Vector& normal,
                               double gamma,
                               Vector& diss,
                               double sigma_l = 0.0,
                               double sigma_r = 0.0)
{
  const int num_eq = ul.Size();
  if (num_eq == 3)
  {
    WintersDissipation1D(ul, ur, normal, gamma, sigma_l, sigma_r, diss);
    return;
  }
  if (num_eq == 4)
  {
    WintersDissipation2D(ul, ur, normal, gamma, sigma_l, sigma_r, diss);
    return;
  }
  MFEM_VERIFY(false, "WintersDissipation supports Euler in 1D or 2D.");
}

inline void EntropyVariableLFDissipation(const Vector& ul,
                                         const Vector& ur,
                                         double gamma,
                                         double scale,
                                         Vector& diss,
                                         double sigma_l = 0.0,
                                         double sigma_r = 0.0,
                                         EntropyJacobianMode mode =
                                             EntropyJacobianMode::FiniteDifference)
{
  const int num_eq = ul.Size();
  MFEM_VERIFY(num_eq == ur.Size(),
              "EntropyVariableLFDissipation requires matching state sizes.");
  const int dim = num_eq - 2;
  MFEM_VERIFY(dim == 1 || dim == 2,
              "EntropyVariableLFDissipation supports dim=1 or 2.");

  Vector w_l(num_eq);
  Vector w_r(num_eq);
  EulerEntropyVariables(ul, dim, gamma, sigma_l, w_l);
  EulerEntropyVariables(ur, dim, gamma, sigma_r, w_r);

  Vector w_jump(num_eq);
  w_jump = w_r;
  w_jump -= w_l;

  Vector u_bar(num_eq);
  double sigma_bar = 0.0;
  BuildMeanState(ul, ur, dim, gamma, sigma_l, sigma_r, u_bar, sigma_bar);

  DenseMatrix J(num_eq);
  if (mode == EntropyJacobianMode::Analytic)
  {
    EntropyJacobianAnalytic(u_bar, dim, gamma, sigma_bar, J);
  }
  else
  {
    EntropyJacobian(u_bar, dim, gamma, sigma_bar, J);
  }

  DenseMatrixInverse invJ(J);
  Vector delta_u(num_eq);
  invJ.Mult(w_jump, delta_u);

  diss.SetSize(num_eq);
  for (int i = 0; i < num_eq; ++i)
  {
    diss(i) = -0.5 * scale * delta_u(i);
  }
}

} // namespace dgsem
} // namespace mfem

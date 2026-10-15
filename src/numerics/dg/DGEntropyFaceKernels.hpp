#pragma once

// Private raw-storage translation of dgsem_entropy_flux.hpp. The CPU oracle
// remains independent: MFEM's DGSEMSurfaceFlux and NonlinearForm assembly.
#include "mfem.hpp"
#include <cmath>
#include <limits>

namespace mfem { namespace dg_igr_detail { namespace packed_entropy {
MFEM_HOST_DEVICE inline double LogMean(double a, double b)
{
  const double diff = b - a;
  const double sum = a + b;
  const double tol = 1.0e-12 * fmax(1.0, fabs(sum));
  if (fabs(diff) < tol)
  {
    return 0.5 * sum;
  }
  return diff / (log(b) - log(a));
}

MFEM_HOST_DEVICE inline double InvLogMean(double a, double b)
{
  const double diff = b - a;
  const double sum = a + b;
  const double tol = 1.0e-12 * fmax(1.0, fabs(sum));
  if (fabs(diff) < tol)
  {
    return 2.0 / sum;
  }
  return (log(b) - log(a)) / diff;
}

MFEM_HOST_DEVICE inline void EulerPrimitive(const real_t* u, int dim,
  double gamma, double sigma, double& rho, double& v1, double& v2,
  double& p, double& rho_e)
{
  rho = u[0]; rho_e = u[dim + 1];
  v1 = u[1] / rho; v2 = u[2] / rho;
  const double v_sq = v1 * v1 + v2 * v2;
  p = (gamma - 1.0) * (rho_e - 0.5 * rho * v_sq) + sigma;
}
MFEM_HOST_DEVICE inline void EulerEntropyVariables(const real_t* u,
                                  int dim,
                                  double gamma,
                                  double sigma,
                                  real_t* w)
{
  double rho, v1, v2, p, rho_e;
  EulerPrimitive(u, dim, gamma, sigma, rho, v1, v2, p, rho_e);
  const double v_sq = v1 * v1 + v2 * v2;
  const double s = log(p) - gamma * log(rho);
  const double rho_p = rho / p;

  w[0] = (gamma - s) / (gamma - 1.0) - 0.5 * rho_p * v_sq;
  w[1] = rho_p * v1;
  if (dim == 2)
  {
    w[2] = rho_p * v2;
    w[3] = -rho_p;
  }
  else
  {
    w[2] = -rho_p;
  }
}

MFEM_HOST_DEVICE inline void RanochaDotN(const real_t* ul, const real_t* ur,
  const real_t* normal, double gamma, double sigma_l, double sigma_r, real_t* flux)
{
  real_t f_dir[4];
  for (int eq = 0; eq < 4; ++eq) { flux[eq] = 0.0; }
  // Both directions use the same states and logarithmic means. Retain each
  // directional expression and the original +0, x, y accumulation order,
  // including work for zero normal components.
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

  for (int d = 0; d < 2; ++d)
  {
    if (d == 0)
    {
      const double f1 = rho_mean * v1_avg;
      const double f2 = f1 * v1_avg + p_avg;
      const double f3 = f1 * v2_avg;
      const double f4 = f1 * (v_sq_avg + inv_rho_p_mean / (gamma - 1.0)) +
                        0.5 * (p_l * v1_r + p_r * v1_l);
      f_dir[0] = f1;
      f_dir[1] = f2;
      f_dir[2] = f3;
      f_dir[3] = f4;
    }
    else
    {
      const double f1 = rho_mean * v2_avg;
      const double f2 = f1 * v1_avg;
      const double f3 = f1 * v2_avg + p_avg;
      const double f4 = f1 * (v_sq_avg + inv_rho_p_mean / (gamma - 1.0)) +
                        0.5 * (p_l * v2_r + p_r * v2_l);
      f_dir[0] = f1;
      f_dir[1] = f2;
      f_dir[2] = f3;
      f_dir[3] = f4;
    }
    for (int eq = 0; eq < 4; ++eq)
    {
      flux[eq] += normal[d] * f_dir[eq];
    }
  }
}
MFEM_HOST_DEVICE inline void WintersDissipation2D(const real_t* ul,
                                 const real_t* ur,
                                 const real_t* normal,
                                 double gamma,
                                 double sigma_l,
                                 double sigma_r,
                                 real_t* diss)
{
  const double norm = sqrt(normal[0] * normal[0] + normal[1] * normal[1]);
  if (norm == 0.0)
  {
    for (int eq = 0; eq < 4; ++eq) { diss[eq] = 0.0; }
    return;
  }
  const double nx = normal[0] / norm;
  const double ny = normal[1] / norm;

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
  const double c_bar = sqrt(gamma * p_avg / rho_log);
  const double v_avg_normal = v1_avg * nx + v2_avg * ny;

  const double lambda1 =
    fabs(v_avg_normal - c_bar) * rho_log / (2.0 * gamma);
  const double lambda2 =
    fabs(v_avg_normal) * rho_log * (gamma - 1.0) / gamma;
  const double lambda3 =
    fabs(v_avg_normal + c_bar) * rho_log / (2.0 * gamma);
  const double lambda4 = fabs(v_avg_normal) * p_avg;

  const double v1_minus_c = v1_avg - c_bar * nx;
  const double v2_minus_c = v2_avg - c_bar * ny;
  const double v1_plus_c = v1_avg + c_bar * nx;
  const double v2_plus_c = v2_avg + c_bar * ny;
  const double v1_tan = v1_avg - v_avg_normal * nx;
  const double v2_tan = v2_avg - v_avg_normal * ny;

  real_t w_l_data[4], w_r_data[4], w_jump_data[4];
  real_t* w_l = w_l_data; real_t* w_r = w_r_data;
  EulerEntropyVariables(ul, /*dim=*/2, gamma, sigma_l, w_l);
  EulerEntropyVariables(ur, /*dim=*/2, gamma, sigma_r, w_r);
  real_t* w_jump = w_jump_data;
  for (int eq = 0; eq < 4; ++eq) { w_jump[eq] = w_r[eq] - w_l[eq]; }

  const double w1_jump = w_jump[0];
  const double w2_jump = w_jump[1];
  const double w3_jump = w_jump[2];
  const double w4_jump = w_jump[3];

  const double v_minus_dot = v1_minus_c * w2_jump + v2_minus_c * w3_jump;
  const double v_plus_dot = v1_plus_c * w2_jump + v2_plus_c * w3_jump;
  const double v_avg_dot = v1_avg * w2_jump + v2_avg * w3_jump;

  const double w1 = lambda1 * (w1_jump + v_minus_dot +
                               (h_bar - c_bar * v_avg_normal) * w4_jump);
  const double w2 = lambda2 * (w1_jump + v_avg_dot + 0.5 * v_sq_bar * w4_jump);
  const double w3 =
    lambda3 * (w1_jump + v_plus_dot + (h_bar + c_bar * v_avg_normal) * w4_jump);

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
    w1 * (h_bar - c_bar * v_avg_normal) + w2 * 0.5 * v_sq_bar +
    w3 * (h_bar + c_bar * v_avg_normal) +
    lambda4 * (v_tan_dot + w4_jump * (v_avg_sq - v_avg_normal * v_avg_normal));

  diss[0] = -0.5 * diss_rho * norm;
  diss[1] = -0.5 * diss_rho_v1 * norm;
  diss[2] = -0.5 * diss_rho_v2 * norm;
  diss[3] = -0.5 * diss_rhoe * norm;
}

MFEM_HOST_DEVICE inline bool Flux(const real_t* ul, const real_t* ur,
  const real_t* normal, double gamma, double sigma_l, double sigma_r, real_t* flux)
{
  double rho, v1, v2, p, energy;
  EulerPrimitive(ul, 2, gamma, sigma_l, rho, v1, v2, p, energy);
  if (!(rho > 0.0 && p > 0.0)) { return false; }
  EulerPrimitive(ur, 2, gamma, sigma_r, rho, v1, v2, p, energy);
  if (!(rho > 0.0 && p > 0.0)) { return false; }
  RanochaDotN(ul, ur, normal, gamma, sigma_l, sigma_r, flux);
  real_t diss[4];
  WintersDissipation2D(ul, ur, normal, gamma, sigma_l, sigma_r, diss);
  for (int eq = 0; eq < 4; ++eq) { flux[eq] += diss[eq]; }
  return true;
}


MFEM_HOST_DEVICE inline real_t EulerSpeed(const real_t* u, const real_t* n,
                                        real_t gamma)
{
  real_t mom2 = 0.0, mn = 0.0, nn = 0.0;
  for (int d = 0; d < 2; ++d)
  {
    mom2 += u[d + 1] * u[d + 1];
    mn += u[d + 1] * n[d];
    nn += n[d] * n[d];
  }
  const real_t kinetic = 0.5 * mom2 / u[0];
  const real_t pressure = (gamma - 1.0) * (u[3] - kinetic);
  return fabs(mn / u[0]) / sqrt(nn) + sqrt(gamma * pressure / u[0]);
}

struct FaceKernel
{
  int dof, nq, sigma_dof;
  real_t gamma;
  const int *e1, *e2, *state_map, *sigma_map;
  const real_t *x, *sigma, *shape1, *shape2, *sigma_shape1, *sigma_shape2, *normal;
  real_t *flux, *speed;

  MFEM_HOST_DEVICE void operator()(int fq) const
  {
    const int f = fq / nq;
    real_t ul[4] = {}, ur[4] = {};
    for (int i = 0; i < dof; ++i)
    {
      for (int eq = 0; eq < 4; ++eq)
      {
        ul[eq] += shape1[fq * dof + i] * x[state_map[(e1[f] * dof + i) * 4 + eq]];
        ur[eq] += shape2[fq * dof + i] * x[state_map[(e2[f] * dof + i) * 4 + eq]];
      }
    }
    real_t sl = 0.0, sr = 0.0;
    if (sigma)
    {
      for (int i = 0; i < sigma_dof; ++i)
      {
        sl += sigma_shape1[fq * sigma_dof + i] * sigma[sigma_map[e1[f] * sigma_dof + i]];
        sr += sigma_shape2[fq * sigma_dof + i] * sigma[sigma_map[e2[f] * sigma_dof + i]];
      }
    }
    real_t* out = flux + 4 * fq;
    for (int eq = 0; eq < 4; ++eq) { out[eq] = 0.0; }
    const real_t* n = normal + 2 * fq;
    bool valid = Flux(ul, ur, n, gamma, sl, sr, out);
    const real_t speed1 = EulerSpeed(ul, n, gamma);
    const real_t speed2 = EulerSpeed(ur, n, gamma);
    const real_t limit = std::numeric_limits<real_t>::max();
    valid = valid && speed1 >= 0.0 && speed1 <= limit && speed2 >= 0.0 && speed2 <= limit;
    for (int eq = 0; eq < 4; ++eq) { valid = valid && out[eq] >= -limit && out[eq] <= limit; }
    speed[fq] = valid ? (speed1 > speed2 ? speed1 : speed2)
                      : std::numeric_limits<real_t>::infinity();
  }
};

// Each element owns its residual entries. Match MFEM's local per-face q sum,
// followed by the ascending mesh-face accumulation, without atomic additions.
struct GatherKernel
{
  int dof, nq;
  const int *offsets, *sides, *state_map;
  const real_t *shape1, *shape2, *weights, *flux;
  real_t* z;
  MFEM_HOST_DEVICE void operator()(int entry) const
  {
    const int eq = entry % 4;
    const int node = entry / 4;
    const int e = node / dof;
    const int i = node % dof;
    real_t sum = 0.0;
    for (int incidence = offsets[e]; incidence < offsets[e + 1]; ++incidence)
    {
      const int side = sides[incidence];
      const int f = side / 2;
      const bool right = side % 2;
      const real_t* shape = right ? shape2 : shape1;
      real_t face_sum = 0.0;
      for (int q = 0; q < nq; ++q)
      {
        const int fq = f * nq + q;
        const real_t weighted_flux = (right ? weights[fq] : -weights[fq]) * flux[fq * 4 + eq];
        face_sum += shape[fq * dof + i] * weighted_flux;
      }
      sum += face_sum;
    }
    z[state_map[entry]] = sum;
  }
};

} } } // namespace mfem::dg_igr_detail::packed_entropy

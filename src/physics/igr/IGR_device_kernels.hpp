// Internal-only helper/kernels for IGR device coefficient path.
// This file is intentionally included from `IGR.cpp` after MFEM backend headers.
#pragma once

#include "mfem/general/annotation.hpp"

namespace dg_igr
{
namespace igr_device
{

struct IGRDeviceCoeffKernel
{
  int nq;
  int ne;
  int dim;
  int num_eq;
  mfem::real_t alpha;
  mfem::real_t rho_floor;
  const mfem::real_t* u;
  const mfem::real_t* grad;
  mfem::real_t* invrho;
  mfem::real_t* ck;
  mfem::real_t* src;

  MFEM_HOST_DEVICE void operator()(int idx) const
  {
    const int e = idx / nq;
    const int q = idx - e * nq;
    const int q_idx = q + nq * e;
    const int u_base = num_eq * q_idx;

    mfem::real_t rho = u[u_base];
    if (rho < rho_floor) { rho = rho_floor; }
    const mfem::real_t inv_rho = 1.0 / rho;
    invrho[e * nq + q] = inv_rho;
    ck[e * nq + q] = alpha * inv_rho;

    mfem::real_t divu = 0.0;
    mfem::real_t trA2 = 0.0;
    mfem::real_t gradu[3][3] = {{0.0, 0.0, 0.0},
                                {0.0, 0.0, 0.0},
                                {0.0, 0.0, 0.0}};

    const mfem::real_t rho2 = rho * rho;
    for (int i = 0; i < dim; ++i)
    {
      const mfem::real_t mom_i = u[u_base + 1 + i];
      for (int j = 0; j < dim; ++j)
      {
        const int grad_base = num_eq * (j + dim * q_idx);
        const mfem::real_t dmij = grad[grad_base + 1 + i];
        const mfem::real_t drhoj = grad[grad_base];
        gradu[i][j] = (rho * dmij - mom_i * drhoj) / rho2;
      }
    }

    for (int i = 0; i < dim; ++i)
    {
      divu += gradu[i][i];
    }
    for (int i = 0; i < dim; ++i)
    {
      for (int j = 0; j < dim; ++j)
      {
        trA2 += gradu[i][j] * gradu[j][i];
      }
    }

    src[e * nq + q] = alpha * (trA2 + divu * divu);
  }
};

struct IGRDeviceInvRhoKernel
{
  int nq;
  int ne;
  int num_eq;
  mfem::real_t rho_floor;
  const mfem::real_t* u;
  mfem::real_t* invrho;

  MFEM_HOST_DEVICE void operator()(int idx) const
  {
    const int e = idx / nq;
    const int q = idx - e * nq;
    const int q_idx = q + nq * e;
    const int u_base = num_eq * q_idx;
    mfem::real_t rho = u[u_base];
    if (rho < rho_floor) { rho = rho_floor; }
    invrho[e * nq + q] = 1.0 / rho;
  }
};

struct IGRDeviceAlphaOverRhoKernel
{
  int nq;
  int ne;
  int num_eq;
  mfem::real_t alpha;
  mfem::real_t rho_floor;
  const mfem::real_t* u;
  mfem::real_t* ck;

  MFEM_HOST_DEVICE void operator()(int idx) const
  {
    const int e = idx / nq;
    const int q = idx - e * nq;
    const int q_idx = q + nq * e;
    const int u_base = num_eq * q_idx;
    mfem::real_t rho = u[u_base];
    if (rho < rho_floor) { rho = rho_floor; }
    ck[e * nq + q] = alpha / rho;
  }
};

struct IGRDeviceSourceKernel
{
  int nq;
  int ne;
  int dim;
  int num_eq;
  mfem::real_t alpha;
  mfem::real_t rho_floor;
  const mfem::real_t* u;
  const mfem::real_t* grad;
  mfem::real_t* src;

  MFEM_HOST_DEVICE void operator()(int idx) const
  {
    const int e = idx / nq;
    const int q = idx - e * nq;
    const int q_idx = q + nq * e;
    const int u_base = num_eq * q_idx;

    mfem::real_t rho = u[u_base];
    if (rho < rho_floor) { rho = rho_floor; }

    mfem::real_t divu = 0.0;
    mfem::real_t trA2 = 0.0;
    mfem::real_t gradu[3][3] = {{0.0, 0.0, 0.0},
                                {0.0, 0.0, 0.0},
                                {0.0, 0.0, 0.0}};

    const mfem::real_t rho2 = rho * rho;
    for (int i = 0; i < dim; ++i)
    {
      const mfem::real_t mom_i = u[u_base + 1 + i];
      for (int j = 0; j < dim; ++j)
      {
        const int grad_base = num_eq * (j + dim * q_idx);
        const mfem::real_t dmij = grad[grad_base + 1 + i];
        const mfem::real_t drhoj = grad[grad_base];
        gradu[i][j] = (rho * dmij - mom_i * drhoj) / rho2;
      }
    }

    for (int i = 0; i < dim; ++i)
    {
      divu += gradu[i][i];
    }
    for (int i = 0; i < dim; ++i)
    {
      for (int j = 0; j < dim; ++j)
      {
        trA2 += gradu[i][j] * gradu[j][i];
      }
    }

    src[e * nq + q] = alpha * (trA2 + divu * divu);
  }
};

struct IGRDeviceFaceCoeffKernel
{
  int nq;
  int nf;
  int num_eq;
  mfem::real_t alpha;
  mfem::real_t rho_floor;
  const mfem::real_t* uface;
  mfem::real_t* ckface;

  MFEM_HOST_DEVICE void operator()(int idx) const
  {
    const int f = idx / nq;
    const int q = idx - f * nq;
    const int q_idx = q + nq * f;
    const int u_base = num_eq * q_idx;
    mfem::real_t rho = uface[u_base];
    if (rho < rho_floor) { rho = rho_floor; }
    ckface[f * nq + q] = alpha / rho;
  }
};

} // namespace igr_device
} // namespace dg_igr

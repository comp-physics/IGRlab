// Internal-only helper/kernels for IGR device coefficient path.
// This file is intentionally included from `IGR.cpp` after MFEM backend headers.
#pragma once

#include "igrlab/compat.hpp"

#include "mfem/general/annotation.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"

namespace igrlab
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

    invrho[e * nq + q] =
      frozen_scalar_igr_algebra::SafeInverseDensity(u[u_base], rho_floor);
    ck[e * nq + q] = alpha * invrho[e * nq + q];
    src[e * nq + q] =
      frozen_scalar_igr_algebra::SourceFromConservativeStateGradient(
        dim,
        u + u_base,
        1,
        grad + num_eq * dim * q_idx,
        1,
        num_eq,
        alpha,
        rho_floor);
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
    invrho[e * nq + q] =
      frozen_scalar_igr_algebra::SafeInverseDensity(u[u_base], rho_floor);
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
    ck[e * nq + q] =
      frozen_scalar_igr_algebra::AlphaOverDensity(alpha, u[u_base], rho_floor);
  }
};

struct IGRDeviceLocalAlphaOverRhoKernel
{
  int nq;
  int ne;
  int num_eq;
  mfem::real_t rho_floor;
  const mfem::real_t* element_alpha;
  const mfem::real_t* u;
  mfem::real_t* ck;

  MFEM_HOST_DEVICE void operator()(int idx) const
  {
    const int e = idx / nq;
    const int q = idx - e * nq;
    const int q_idx = q + nq * e;
    const int u_base = num_eq * q_idx;
    ck[e * nq + q] = frozen_scalar_igr_algebra::AlphaOverDensity(
      element_alpha[e], u[u_base], rho_floor);
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

    src[e * nq + q] =
      frozen_scalar_igr_algebra::SourceFromConservativeStateGradient(
        dim,
        u + u_base,
        1,
        grad + num_eq * dim * q_idx,
        1,
        num_eq,
        alpha,
        rho_floor);
  }
};

struct IGRDeviceLocalSourceKernel
{
  int nq;
  int ne;
  int dim;
  int num_eq;
  mfem::real_t rho_floor;
  const mfem::real_t* element_alpha;
  const mfem::real_t* u;
  const mfem::real_t* grad;
  mfem::real_t* src;

  MFEM_HOST_DEVICE void operator()(int idx) const
  {
    const int e = idx / nq;
    const int q = idx - e * nq;
    const int q_idx = q + nq * e;
    const int u_base = num_eq * q_idx;

    src[e * nq + q] =
      frozen_scalar_igr_algebra::SourceFromConservativeStateGradient(
        dim,
        u + u_base,
        1,
        grad + num_eq * dim * q_idx,
        1,
        num_eq,
        element_alpha[e],
        rho_floor);
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
    ckface[f * nq + q] =
      frozen_scalar_igr_algebra::AlphaOverDensity(alpha,
                                                  uface[u_base],
                                                  rho_floor);
  }
};

} // namespace igr_device
} // namespace igrlab

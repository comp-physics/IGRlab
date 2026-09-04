// Internal-only helper/kernels for IGR device coefficient path.
// This file is intentionally included from `IGR.cpp` after MFEM backend headers.
#pragma once

#include "mfem/general/annotation.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"

namespace mfem
{
class Vector;
}

namespace dg_igr
{
namespace igr_device
{

void LaunchExactElementBlockJacobiApply(
  int elements,
  int block_size,
  const mfem::Vector& inverse_blocks,
  const mfem::Vector& x,
  mfem::Vector& y);

void LaunchElementMeanProjection(int elements,
                                 int block_size,
                                 const mfem::Vector& x,
                                 mfem::Vector& y);

void LaunchElementConstantRestriction(int elements,
                                      int block_size,
                                      const mfem::Vector& x,
                                      mfem::Vector& coarse);

void LaunchElementConstantProlongation(int elements,
                                       int block_size,
                                       const mfem::Vector& coarse,
                                       mfem::Vector& y);

struct IGRExactElementBlockJacobiKernel
{
  int block_size;
  const mfem::real_t* inverse_blocks;
  const mfem::real_t* input;
  mfem::real_t* output;

  MFEM_HOST_DEVICE void operator()(int entry) const
  {
    const int element = entry / block_size;
    const int row = entry - element * block_size;
    mfem::real_t value = 0.0;
    for (int column = 0; column < block_size; ++column)
    {
      value += inverse_blocks[row + block_size *
                                      (column + block_size * element)] *
               input[column + block_size * element];
    }
    output[row + block_size * element] = value;
  }
};

struct IGRElementMeanProjectionKernel
{
  int block_size;
  const mfem::real_t* input;
  mfem::real_t* output;

  MFEM_HOST_DEVICE void operator()(int element) const
  {
    mfem::real_t mean = 0.0;
    for (int local = 0; local < block_size; ++local)
    {
      mean += input[local + block_size * element];
    }
    mean /= static_cast<mfem::real_t>(block_size);
    for (int local = 0; local < block_size; ++local)
    {
      output[local + block_size * element] =
        input[local + block_size * element] - mean;
    }
  }
};

struct IGRElementConstantRestrictionKernel
{
  int block_size;
  const mfem::real_t* input;
  mfem::real_t* coarse;

  MFEM_HOST_DEVICE void operator()(int element) const
  {
    mfem::real_t value = 0.0;
    for (int local = 0; local < block_size; ++local)
    {
      value += input[local + block_size * element];
    }
    coarse[element] = value;
  }
};

struct IGRElementConstantProlongationKernel
{
  int block_size;
  const mfem::real_t* coarse;
  mfem::real_t* output;

  MFEM_HOST_DEVICE void operator()(int entry) const
  {
    output[entry] = coarse[entry / block_size];
  }
};

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
} // namespace dg_igr

// Internal inverse-mass kernels; included inside namespace mfem by mass.cpp.
#pragma once

namespace
{

// Component reuse does not depend on Euler flux algebra or spatial dimension.
// Keep the bounded component variants separate from the legacy kernel type.
template <int Components>
struct PackedInverseMassRowReuseKernel
{
  static_assert(Components >= 2 && Components <= 5,
                "Inverse-mass row reuse requires two through five components.");
  int dof = 0;
  const int* map = nullptr;
  const real_t* minv = nullptr;
  const real_t* zin = nullptr;
  real_t* yout = nullptr;

  MFEM_HOST_DEVICE void operator()(int entry) const
  {
    const int i = entry % dof, e = entry / dof;
    const int voff = e * dof * Components, moff = e * dof * dof;
    real_t sum[Components] = {};
    for (int j = 0; j < dof; ++j)
    {
      const real_t weight = minv[moff + j * dof + i];
#if defined(__CUDACC__)
#pragma unroll
#endif
      for (int eq = 0; eq < Components; ++eq)
      {
        // Each component retains the original ascending-j FP64 dot product.
        sum[eq] += weight * zin[map[voff + j * Components + eq]];
      }
    }
#if defined(__CUDACC__)
#pragma unroll
#endif
    for (int eq = 0; eq < Components; ++eq)
    {
      yout[map[voff + i * Components + eq]] = sum[eq];
    }
  }
};

} // namespace

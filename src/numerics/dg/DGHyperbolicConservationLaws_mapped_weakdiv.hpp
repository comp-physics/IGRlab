#pragma once

// Private mapped weak-divergence storage/kernels. Included inside mfem after
// its ordinary host/CUDA backend headers. No state interpolation precedes flux.
namespace dg_igr_detail
{

struct MappedWeakDivTensorData
{
  int n = 0, q = 0, elements = 0;
  long space_sequence = -1, mesh_sequence = -1, nodes_sequence = -1;
  Array<int> lex_to_local;
  Vector basis, derivative;
  // [element][qy*q+qx][reference derivative][physical flux direction].
  // These are the original GradientIntegrator quadrature weights times Jadj.
  Vector weighted_adjugate;
};

#ifdef DG_IGR_HAS_ACCEL_FORALL
struct MappedWeakDivTensorKernel2D
{
  static constexpr int MaxN = 8, MaxQ = 10;
  int n = 0, q = 0;
  const int* lex_to_local = nullptr;
  const int* state_map = nullptr;
  const real_t* basis = nullptr;
  const real_t* derivative = nullptr;
  const real_t* geometry = nullptr;
  const real_t* nodal_flux = nullptr;
  real_t* residual = nullptr;

  MFEM_HOST_DEVICE void operator()(int e) const
  {
    MFEM_SHARED real_t B[MaxN][MaxQ], G[MaxN][MaxQ];
    MFEM_SHARED real_t s0[MaxQ][MaxQ], s1[MaxQ][MaxQ];
    MFEM_SHARED real_t t0[MaxQ][MaxQ], t1[MaxQ][MaxQ];
    MFEM_FOREACH_THREAD(i, y, n)
    {
      MFEM_FOREACH_THREAD(k, x, q)
      {
        B[i][k] = basis[k + q * i];
        G[i][k] = derivative[k + q * i];
      }
    }
    MFEM_SYNC_THREAD;
    const int dof = n * n, flux_stride = 2 * dof;
    // One block owns each cell's four equations. Each directional packet is
    // the unchanged FP64 nonlinear flux evaluated at the original state nodes.
    for (int c = 0; c < 4; ++c)
    {
      MFEM_FOREACH_THREAD(jy, y, n)
      {
        MFEM_FOREACH_THREAD(jx, x, n)
        {
          const int local = lex_to_local[jx + n * jy];
          const int offset = (e * 4 + c) * flux_stride + 2 * local;
          s0[jy][jx] = nodal_flux[offset];
          s1[jy][jx] = nodal_flux[offset + 1];
        }
      }
      MFEM_SYNC_THREAD;
      // Interpolate the nodal flux in x, then y, at the native quadrature.
      MFEM_FOREACH_THREAD(jy, y, n)
      {
        MFEM_FOREACH_THREAD(qx, x, q)
        {
          real_t fx = 0.0, fy = 0.0;
          for (int jx = 0; jx < n; ++jx)
          {
            fx += B[jx][qx] * s0[jy][jx];
            fy += B[jx][qx] * s1[jy][jx];
          }
          t0[jy][qx] = fx; t1[jy][qx] = fy;
        }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy, y, q)
      {
        MFEM_FOREACH_THREAD(qx, x, q)
        {
          real_t fx = 0.0, fy = 0.0;
          for (int jy = 0; jy < n; ++jy)
          {
            fx += B[jy][qy] * t0[jy][qx];
            fy += B[jy][qy] * t1[jy][qx];
          }
          const int metric = 4 * (e * q * q + qx + q * qy);
          s0[qy][qx] = geometry[metric] * fx + geometry[metric + 1] * fy;
          s1[qy][qx] = geometry[metric + 2] * fx + geometry[metric + 3] * fy;
        }
      }
      MFEM_SYNC_THREAD;
      // Apply test B in y to xi flux and test G in y to eta flux.
      MFEM_FOREACH_THREAD(iy, y, n)
      {
        MFEM_FOREACH_THREAD(qx, x, q)
        {
          real_t xi = 0.0, eta = 0.0;
          for (int qy = 0; qy < q; ++qy)
          {
            xi += B[iy][qy] * s0[qy][qx];
            eta += G[iy][qy] * s1[qy][qx];
          }
          t0[iy][qx] = xi; t1[iy][qx] = eta;
        }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(iy, y, n)
      {
        MFEM_FOREACH_THREAD(ix, x, n)
        {
          real_t value = 0.0;
          for (int qx = 0; qx < q; ++qx)
          {
            value += G[ix][qx] * t0[iy][qx] + B[ix][qx] * t1[iy][qx];
          }
          const int local = lex_to_local[ix + n * iy];
          // Retain the completed boundary/interior face contribution.
          residual[state_map[(e * dof + local) * 4 + c]] += value;
        }
      }
      MFEM_SYNC_THREAD;
    }
  }
};
#endif

} // namespace dg_igr_detail

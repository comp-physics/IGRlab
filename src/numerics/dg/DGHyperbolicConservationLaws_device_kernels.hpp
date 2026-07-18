// Internal-only helper/kernels for DGHyperbolicConservationLaws device path.
// This file is intentionally included from
// `DGHyperbolicConservationLaws.cpp` after MFEM/CUDA/HIP backend headers.
#pragma once

namespace
{

enum class DeviceBoundaryType : int
{
  Unsupported = 0,
  SupersonicInlet = 1,
  SupersonicOutlet = 2,
  SubsonicOutletPressure = 3,
  NonReflectingOutlet = 4,
  SubsonicInlet = 5,
  FarField = 6,
  SlipWall = 7
};

#ifdef DG_IGR_HAS_ACCEL_FORALL
constexpr real_t kBoundaryRhoFloor = 1.0e-12;
constexpr real_t kBoundaryPFloor = 1.0e-12;

struct PrimitiveNormalState_
{
  real_t rho = 1.0;
  real_t p = 1.0;
  real_t un = 0.0;
  real_t ut = 0.0;
  real_t c = 1.0;
};

MFEM_HOST_DEVICE inline void BuildNormalTangential_(const real_t* normal,
                                                     int dim,
                                                     real_t* nhat,
                                                     real_t* that)
{
  real_t nrm2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    nhat[d] = normal[d];
    nrm2 += normal[d] * normal[d];
    that[d] = 0.0;
  }
  const real_t nrm = sqrt(nrm2);
  if (nrm > 0.0)
  {
    const real_t inv_nrm = 1.0 / nrm;
    for (int d = 0; d < dim; ++d)
    {
      nhat[d] *= inv_nrm;
    }
  }
  else
  {
    for (int d = 0; d < dim; ++d)
    {
      nhat[d] = 0.0;
    }
    nhat[0] = 1.0;
  }

  if (dim == 2)
  {
    that[0] = -nhat[1];
    that[1] = nhat[0];
  }
}

MFEM_HOST_DEVICE inline PrimitiveNormalState_ ConservativeToPrimitiveNormal_(
  const real_t* U,
  int dim,
  real_t gamma,
  const real_t* nhat,
  const real_t* that)
{
  PrimitiveNormalState_ out;
  const real_t rho = fmax(U[0], kBoundaryRhoFloor);
  real_t vel[3] = {0.0, 0.0, 0.0};
  real_t vel2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    vel[d] = U[1 + d] / rho;
    vel2 += vel[d] * vel[d];
  }

  const real_t E = U[1 + dim] / rho;
  const real_t kinetic = 0.5 * vel2;
  const real_t p = fmax((gamma - 1.0) * rho * (E - kinetic), kBoundaryPFloor);
  const real_t c = sqrt(fmax(gamma * p / rho, 0.0));

  out.rho = rho;
  out.p = p;
  out.c = c;
  out.un = 0.0;
  out.ut = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    out.un += vel[d] * nhat[d];
  }
  if (dim == 2)
  {
    out.ut = vel[0] * that[0] + vel[1] * that[1];
  }
  return out;
}

MFEM_HOST_DEVICE inline void PrimitiveNormalToConservative_(
  const PrimitiveNormalState_& prim,
  int dim,
  real_t gamma,
  const real_t* nhat,
  const real_t* that,
  real_t* U)
{
  real_t vel[3] = {0.0, 0.0, 0.0};
  if (dim == 1)
  {
    vel[0] = prim.un;
  }
  else
  {
    vel[0] = prim.un * nhat[0] + prim.ut * that[0];
    vel[1] = prim.un * nhat[1] + prim.ut * that[1];
  }

  U[0] = prim.rho;
  real_t vel2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    U[1 + d] = prim.rho * vel[d];
    vel2 += vel[d] * vel[d];
  }
  const real_t E = prim.p / ((gamma - 1.0) * prim.rho) + 0.5 * vel2;
  U[1 + dim] = prim.rho * E;
}

MFEM_HOST_DEVICE inline real_t EntropyLike_(real_t p, real_t rho, real_t gamma)
{
  return p / pow(fmax(rho, kBoundaryRhoFloor), gamma);
}

MFEM_HOST_DEVICE inline PrimitiveNormalState_ SubsonicOutletPressureClosure_(
  const PrimitiveNormalState_& inner,
  real_t gamma,
  real_t p_out)
{
  PrimitiveNormalState_ out = inner;
  if (inner.un >= inner.c || inner.un <= -inner.c)
  {
    return out;
  }

  const real_t g1 = gamma - 1.0;
  const real_t s = EntropyLike_(inner.p, inner.rho, gamma);
  const real_t p = fmax(p_out, kBoundaryPFloor);
  const real_t rho = pow(p / s, 1.0 / gamma);
  const real_t c = sqrt(fmax(gamma * p / rho, 0.0));
  const real_t j_plus = inner.un + 2.0 * inner.c / g1;
  const real_t un = j_plus - 2.0 * c / g1;

  out.rho = fmax(rho, kBoundaryRhoFloor);
  out.p = p;
  out.c = c;
  out.un = un;
  out.ut = inner.ut;
  return out;
}

MFEM_HOST_DEVICE inline PrimitiveNormalState_ NonReflectingOutletClosure_(
  const PrimitiveNormalState_& inner,
  const PrimitiveNormalState_& ref,
  real_t gamma)
{
  PrimitiveNormalState_ out = inner;
  if (inner.un >= inner.c || inner.un <= -inner.c)
  {
    return out;
  }

  const real_t g1 = gamma - 1.0;
  const real_t j_plus = inner.un + 2.0 * inner.c / g1;
  const real_t j_minus_ref = ref.un - 2.0 * ref.c / g1;
  const real_t un = 0.5 * (j_plus + j_minus_ref);
  const real_t c = 0.25 * g1 * (j_plus - j_minus_ref);
  if (!(c > 0.0))
  {
    return out;
  }

  const real_t s_ref = EntropyLike_(ref.p, ref.rho, gamma);
  const real_t rho =
    pow((c * c) / fmax(gamma * s_ref, kBoundaryPFloor), 1.0 / g1);
  const real_t p = s_ref * pow(fmax(rho, kBoundaryRhoFloor), gamma);

  out.rho = fmax(rho, kBoundaryRhoFloor);
  out.p = fmax(p, kBoundaryPFloor);
  out.c = c;
  out.un = un;
  out.ut = inner.ut;
  return out;
}

MFEM_HOST_DEVICE inline PrimitiveNormalState_ SubsonicInletClosure_(
  const PrimitiveNormalState_& inner,
  const PrimitiveNormalState_& target,
  real_t gamma)
{
  PrimitiveNormalState_ out = target;
  if (inner.un <= -inner.c)
  {
    return out;
  }
  if (inner.un >= inner.c)
  {
    return inner;
  }

  const real_t g1 = gamma - 1.0;
  const real_t j_plus_inner = inner.un + 2.0 * inner.c / g1;
  const real_t j_minus_target = target.un - 2.0 * target.c / g1;
  const real_t un = 0.5 * (j_plus_inner + j_minus_target);
  const real_t c = 0.25 * g1 * (j_plus_inner - j_minus_target);
  if (!(c > 0.0))
  {
    return target;
  }

  const real_t s_target = EntropyLike_(target.p, target.rho, gamma);
  const real_t rho =
    pow((c * c) / fmax(gamma * s_target, kBoundaryPFloor), 1.0 / g1);
  const real_t p = s_target * pow(fmax(rho, kBoundaryRhoFloor), gamma);

  out.rho = fmax(rho, kBoundaryRhoFloor);
  out.p = fmax(p, kBoundaryPFloor);
  out.c = c;
  out.un = un;
  out.ut = target.ut;
  return out;
}

MFEM_HOST_DEVICE inline bool EvalBoundaryGhostEuler_(int bdr_type,
                                                      int dim,
                                                      int neq,
                                                      real_t gamma,
                                                      real_t p_out,
                                                      const real_t* state_ref,
                                                      const real_t* normal,
                                                      const real_t* u_minus,
                                                      real_t* u_plus)
{
  if (neq != dim + 2)
  {
    return false;
  }

  if (bdr_type == static_cast<int>(DeviceBoundaryType::SupersonicInlet))
  {
    for (int eq = 0; eq < neq; ++eq)
    {
      u_plus[eq] = state_ref[eq];
    }
    return true;
  }
  if (bdr_type == static_cast<int>(DeviceBoundaryType::SupersonicOutlet))
  {
    for (int eq = 0; eq < neq; ++eq)
    {
      u_plus[eq] = u_minus[eq];
    }
    return true;
  }
  if (bdr_type == static_cast<int>(DeviceBoundaryType::SlipWall))
  {
    for (int eq = 0; eq < neq; ++eq)
    {
      u_plus[eq] = u_minus[eq];
    }
    const real_t rho = fmax(u_minus[0], kBoundaryRhoFloor);
    real_t nhat[3] = {0.0, 0.0, 0.0};
    real_t that[3] = {0.0, 0.0, 0.0};
    BuildNormalTangential_(normal, dim, nhat, that);
    real_t un = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      un += (u_minus[1 + d] / rho) * nhat[d];
    }
    for (int d = 0; d < dim; ++d)
    {
      const real_t uprime = (u_minus[1 + d] / rho) - 2.0 * un * nhat[d];
      u_plus[1 + d] = rho * uprime;
    }
    return true;
  }

  real_t nhat[3] = {0.0, 0.0, 0.0};
  real_t that[3] = {0.0, 0.0, 0.0};
  BuildNormalTangential_(normal, dim, nhat, that);

  const PrimitiveNormalState_ inner =
    ConservativeToPrimitiveNormal_(u_minus, dim, gamma, nhat, that);

  PrimitiveNormalState_ bnd = inner;
  if (bdr_type ==
      static_cast<int>(DeviceBoundaryType::SubsonicOutletPressure))
  {
    bnd = SubsonicOutletPressureClosure_(inner, gamma, p_out);
  }
  else if (bdr_type ==
           static_cast<int>(DeviceBoundaryType::NonReflectingOutlet))
  {
    const PrimitiveNormalState_ ref =
      ConservativeToPrimitiveNormal_(state_ref, dim, gamma, nhat, that);
    bnd = NonReflectingOutletClosure_(inner, ref, gamma);
  }
  else if (bdr_type == static_cast<int>(DeviceBoundaryType::SubsonicInlet))
  {
    const PrimitiveNormalState_ target =
      ConservativeToPrimitiveNormal_(state_ref, dim, gamma, nhat, that);
    bnd = SubsonicInletClosure_(inner, target, gamma);
  }
  else if (bdr_type == static_cast<int>(DeviceBoundaryType::FarField))
  {
    const PrimitiveNormalState_ far =
      ConservativeToPrimitiveNormal_(state_ref, dim, gamma, nhat, that);
    if (inner.un <= -inner.c)
    {
      for (int eq = 0; eq < neq; ++eq)
      {
        u_plus[eq] = state_ref[eq];
      }
      return true;
    }
    if (inner.un >= inner.c)
    {
      for (int eq = 0; eq < neq; ++eq)
      {
        u_plus[eq] = u_minus[eq];
      }
      return true;
    }
    bnd = (inner.un < 0.0) ? SubsonicInletClosure_(inner, far, gamma)
                           : NonReflectingOutletClosure_(inner, far, gamma);
  }
  else
  {
    return false;
  }

  PrimitiveNormalToConservative_(bnd, dim, gamma, nhat, that, u_plus);
  return true;
}

MFEM_HOST_DEVICE inline void ComputeEulerFluxDotN_(const real_t* state,
                                                    const real_t* normal,
                                                    int dim,
                                                    real_t gamma,
                                                    real_t* flux_dot_n,
                                                    real_t& speed)
{
  const real_t density = state[0];
  const int energy_eq = 1 + dim;
  real_t mom2 = 0.0;
  real_t normal_mom = 0.0;
  real_t normal_norm2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const real_t md = state[1 + d];
    mom2 += md * md;
    normal_mom += md * normal[d];
    normal_norm2 += normal[d] * normal[d];
  }
  const real_t energy = state[energy_eq];
  const real_t kinetic = 0.5 * mom2 / density;
  const real_t pressure = (gamma - 1.0) * (energy - kinetic);
  const real_t normal_velocity = normal_mom / density;

  flux_dot_n[0] = normal_mom;
  for (int d = 0; d < dim; ++d)
  {
    flux_dot_n[1 + d] = normal_velocity * state[1 + d] + pressure * normal[d];
  }
  flux_dot_n[energy_eq] = normal_velocity * (energy + pressure);

  const real_t sound = sqrt(gamma * pressure / density);
  const real_t normal_norm = sqrt(normal_norm2);
  speed = fabs(normal_velocity) / normal_norm + sound;
}

MFEM_HOST_DEVICE inline void AddSigmaDotNTerms_(const real_t* state,
                                                 const real_t* normal,
                                                 int dim,
                                                 real_t sigma,
                                                 real_t rho_floor,
                                                 real_t* flux_dot_n)
{
  const int energy_eq = 1 + dim;
  for (int d = 0; d < dim; ++d)
  {
    flux_dot_n[1 + d] += sigma * normal[d];
  }
  const real_t rho = fmax(state[0], rho_floor);
  real_t un = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    un += state[1 + d] * normal[d];
  }
  flux_dot_n[energy_eq] += sigma * (un / rho);
}

struct PackedInverseMassKernel
{
  int dof = 0;
  int neq = 0;
  int ldof = 0;
  const int* map = nullptr;
  const real_t* minv = nullptr;
  const real_t* zin = nullptr;
  real_t* yout = nullptr;

  MFEM_HOST_DEVICE void operator()(int e) const
  {
    const int voff = e * ldof;
    const int moff = e * dof * dof;
    for (int eq = 0; eq < neq; ++eq)
    {
      for (int i = 0; i < dof; ++i)
      {
        real_t sum = 0.0;
        for (int j = 0; j < dof; ++j)
        {
          const int zj = map[voff + j * neq + eq];
          sum += minv[moff + j * dof + i] * zin[zj];
        }
        const int yi = map[voff + i * neq + eq];
        yout[yi] = sum;
      }
    }
  }
};

struct PackedWeakDivEulerKernel
{
  int dof = 0;
  int neq = 0;
  int dim = 0;
  int ldof = 0;
  int sigma_ldof = 0;
  real_t gamma = 1.4;
  real_t sigma_rho_floor = 1.0e-12;
  bool use_sigma = false;
  const int* state_map = nullptr;
  const real_t* weakdiv = nullptr;
  const real_t* x = nullptr;
  real_t* z = nullptr;
  const int* sigma_map = nullptr;
  const real_t* sigma_interp = nullptr;
  const real_t* sigma_vals = nullptr;

  MFEM_HOST_DEVICE void operator()(int e) const
  {
    const int state_off = e * ldof;
    const int weak_off = e * dof * dof * dim;
    const int sigma_map_off = e * sigma_ldof;
    const int sigma_interp_off = e * dof * sigma_ldof;

    const int energy_eq = 1 + dim;
    const real_t gm1 = gamma - 1.0;

    for (int j = 0; j < dof; ++j)
    {
      const int xj0 = state_map[state_off + j * neq + 0];
      const real_t rho = x[xj0];
      const real_t inv_rho = 1.0 / rho;

      real_t mom[3] = {0.0, 0.0, 0.0};
      real_t mom2 = 0.0;
      for (int d = 0; d < dim; ++d)
      {
        const int xjd = state_map[state_off + j * neq + (1 + d)];
        mom[d] = x[xjd];
        mom2 += mom[d] * mom[d];
      }
      const real_t energy = x[state_map[state_off + j * neq + energy_eq]];
      const real_t kinetic = 0.5 * mom2 * inv_rho;
      const real_t pressure = gm1 * (energy - kinetic);
      const real_t H = (energy + pressure) * inv_rho;

      real_t sigma_j = 0.0;
      if (use_sigma)
      {
        const int interp_row_off = sigma_interp_off + j * sigma_ldof;
        for (int k = 0; k < sigma_ldof; ++k)
        {
          const int sk = sigma_map[sigma_map_off + k];
          sigma_j += sigma_interp[interp_row_off + k] * sigma_vals[sk];
        }
      }
      const real_t inv_rho_sigma =
        (use_sigma ? 1.0 / fmax(rho, sigma_rho_floor) : 0.0);

      for (int i = 0; i < dof; ++i)
      {
        const int wbase = weak_off + i * (dof * dim) + j * dim;

        // Continuity equation.
        real_t sum_rho = 0.0;
        for (int d = 0; d < dim; ++d)
        {
          sum_rho += weakdiv[wbase + d] * mom[d];
        }
        z[state_map[state_off + i * neq + 0]] += sum_rho;

        // Momentum equations.
        for (int m = 0; m < dim; ++m)
        {
          real_t sum_m = 0.0;
          for (int d = 0; d < dim; ++d)
          {
            real_t flux_md = mom[m] * mom[d] * inv_rho;
            if (m == d)
            {
              flux_md += pressure;
              if (use_sigma)
              {
                flux_md += sigma_j;
              }
            }
            sum_m += weakdiv[wbase + d] * flux_md;
          }
          z[state_map[state_off + i * neq + (1 + m)]] += sum_m;
        }

        // Energy equation.
        real_t sum_e = 0.0;
        for (int d = 0; d < dim; ++d)
        {
          real_t flux_ed = mom[d] * H;
          if (use_sigma)
          {
            flux_ed += sigma_j * mom[d] * inv_rho_sigma;
          }
          sum_e += weakdiv[wbase + d] * flux_ed;
        }
        z[state_map[state_off + i * neq + energy_eq]] += sum_e;
      }
    }
  }
};

struct PackedInteriorFaceEulerRusanovKernel
{
  int dof = 0;
  int neq = 0;
  int dim = 0;
  int nq = 0;
  int ldof = 0;
  int sigma_ldof = 0;
  real_t gamma = 1.4;
  real_t sigma_rho_floor = 1.0e-12;
  bool use_sigma = false;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
  const int* state_map = nullptr;
  const int* face_e1 = nullptr;
  const int* face_e2 = nullptr;
  const real_t* shape1 = nullptr;
  const real_t* shape2 = nullptr;
  const real_t* normals = nullptr;
  const real_t* weights = nullptr;
  const real_t* x = nullptr;
  real_t* z = nullptr;
  real_t* face_speed = nullptr;
  const int* sigma_map = nullptr;
  const real_t* sigma_vals = nullptr;
  const real_t* sigma_shape1 = nullptr;
  const real_t* sigma_shape2 = nullptr;

  MFEM_HOST_DEVICE void operator()(int f) const
  {
    const int e1 = face_e1[f];
    const int e2 = face_e2[f];
    const int state_off_1 = e1 * ldof;
    const int state_off_2 = e2 * ldof;

    real_t u1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t u2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux_hat[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t normal[3] = {0.0, 0.0, 0.0};

    real_t speed_max = 0.0;
    for (int q = 0; q < nq; ++q)
    {
      const int face_q = f * nq + q;
      const int shape_off = face_q * dof;
      const int normal_off = face_q * dim;

      for (int d = 0; d < dim; ++d)
      {
        normal[d] = normals[normal_off + d];
      }

      for (int eq = 0; eq < neq; ++eq)
      {
        real_t lhs = 0.0;
        real_t rhs = 0.0;
        for (int i = 0; i < dof; ++i)
        {
          const real_t s1 = shape1[shape_off + i];
          const real_t s2 = shape2[shape_off + i];
          lhs += s1 * x[state_map[state_off_1 + i * neq + eq]];
          rhs += s2 * x[state_map[state_off_2 + i * neq + eq]];
        }
        u1[eq] = lhs;
        u2[eq] = rhs;
      }

      real_t sigma_1 = 0.0;
      real_t sigma_2 = 0.0;
      if (use_sigma)
      {
        const int sigma_shape_off = face_q * sigma_ldof;
        const int sigma_map_off_1 = e1 * sigma_ldof;
        const int sigma_map_off_2 = e2 * sigma_ldof;
        for (int k = 0; k < sigma_ldof; ++k)
        {
          sigma_1 += sigma_shape1[sigma_shape_off + k] *
                     sigma_vals[sigma_map[sigma_map_off_1 + k]];
          sigma_2 += sigma_shape2[sigma_shape_off + k] *
                     sigma_vals[sigma_map[sigma_map_off_2 + k]];
        }
      }

      real_t speed1 = 0.0;
      real_t speed2 = 0.0;
      ComputeEulerFluxDotN_(u1, normal, dim, gamma, flux1, speed1);
      ComputeEulerFluxDotN_(u2, normal, dim, gamma, flux2, speed2);

      real_t cfl_speed1 = speed1;
      real_t cfl_speed2 = speed2;
      real_t dissipation_speed1 = speed1;
      real_t dissipation_speed2 = speed2;
      if (use_sigma)
      {
        AddSigmaDotNTerms_(
          u1, normal, dim, sigma_1, sigma_rho_floor, flux1);
        AddSigmaDotNTerms_(
          u2, normal, dim, sigma_2, sigma_rho_floor, flux2);
        const real_t igr_speed1 = IGRFluxCharacteristicSpeed(
          speed1, u1[0], sigma_1, sigma_rho_floor);
        const real_t igr_speed2 = IGRFluxCharacteristicSpeed(
          speed2, u2[0], sigma_2, sigma_rho_floor);
        cfl_speed1 = sigma_aware_cfl ? igr_speed1 : speed1;
        cfl_speed2 = sigma_aware_cfl ? igr_speed2 : speed2;
        dissipation_speed1 = sigma_aware_rusanov ? igr_speed1 : speed1;
        dissipation_speed2 = sigma_aware_rusanov ? igr_speed2 : speed2;
      }

      real_t normal_norm2 = 0.0;
      for (int d = 0; d < dim; ++d)
      {
        normal_norm2 += normal[d] * normal[d];
      }
      const real_t scaled_max_speed =
        fmax(dissipation_speed1, dissipation_speed2) * sqrt(normal_norm2);
      speed_max = fmax(speed_max, fmax(cfl_speed1, cfl_speed2));

      for (int eq = 0; eq < neq; ++eq)
      {
        flux_hat[eq] = 0.5 * (scaled_max_speed * (u1[eq] - u2[eq]) + flux1[eq] +
                              flux2[eq]);
      }

      const real_t wq = weights[face_q];
      for (int i = 0; i < dof; ++i)
      {
        const real_t c1 = -wq * shape1[shape_off + i];
        const real_t c2 = +wq * shape2[shape_off + i];
        for (int eq = 0; eq < neq; ++eq)
        {
          const int zi1 = state_map[state_off_1 + i * neq + eq];
          const int zi2 = state_map[state_off_2 + i * neq + eq];
          AtomicAdd(z[zi1], c1 * flux_hat[eq]);
          AtomicAdd(z[zi2], c2 * flux_hat[eq]);
        }
      }
    }
    face_speed[f] = speed_max;
  }
};

struct PackedSharedFaceEulerRusanovKernel
{
  int dof = 0;
  int neq = 0;
  int dim = 0;
  int nq = 0;
  int ldof = 0;
  int sigma_ldof = 0;
  real_t gamma = 1.4;
  real_t sigma_rho_floor = 1.0e-12;
  bool use_sigma = false;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
  const int* state_map = nullptr;
  const int* face_e1 = nullptr;
  const int* face_nbr_vdof_map = nullptr;
  const real_t* shape1 = nullptr;
  const real_t* shape2 = nullptr;
  const real_t* normals = nullptr;
  const real_t* weights = nullptr;
  const real_t* x = nullptr;
  const real_t* x_nbr = nullptr;
  real_t* z = nullptr;
  real_t* face_speed = nullptr;
  const int* sigma_map = nullptr;
  const real_t* sigma_vals = nullptr;
  const int* sigma_nbr_dof_map = nullptr;
  const real_t* sigma_nbr_vals = nullptr;
  const real_t* sigma_shape1 = nullptr;
  const real_t* sigma_shape2 = nullptr;

  MFEM_HOST_DEVICE void operator()(int f) const
  {
    const int e1 = face_e1[f];
    const int state_off_1 = e1 * ldof;
    const int nbr_map_off = f * ldof;

    real_t u1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t u2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux_hat[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t normal[3] = {0.0, 0.0, 0.0};

    real_t speed_max = 0.0;
    for (int q = 0; q < nq; ++q)
    {
      const int face_q = f * nq + q;
      const int shape_off = face_q * dof;
      const int normal_off = face_q * dim;

      for (int d = 0; d < dim; ++d)
      {
        normal[d] = normals[normal_off + d];
      }

      for (int eq = 0; eq < neq; ++eq)
      {
        real_t lhs = 0.0;
        real_t rhs = 0.0;
        for (int i = 0; i < dof; ++i)
        {
          const real_t s1 = shape1[shape_off + i];
          const real_t s2 = shape2[shape_off + i];
          const int vd_nbr = face_nbr_vdof_map[nbr_map_off + i * neq + eq];
          const real_t u2_i = (vd_nbr >= 0) ? x_nbr[vd_nbr]
                                            : -x_nbr[-1 - vd_nbr];
          lhs += s1 * x[state_map[state_off_1 + i * neq + eq]];
          rhs += s2 * u2_i;
        }
        u1[eq] = lhs;
        u2[eq] = rhs;
      }

      real_t sigma_1 = 0.0;
      real_t sigma_2 = 0.0;
      if (use_sigma)
      {
        const int sigma_shape_off = face_q * sigma_ldof;
        const int sigma_map_off_1 = e1 * sigma_ldof;
        const int sigma_map_off_2 = f * sigma_ldof;
        for (int k = 0; k < sigma_ldof; ++k)
        {
          const int sigma_nbr_d = sigma_nbr_dof_map[sigma_map_off_2 + k];
          const real_t sigma_nbr =
            (sigma_nbr_d >= 0) ? sigma_nbr_vals[sigma_nbr_d]
                               : -sigma_nbr_vals[-1 - sigma_nbr_d];
          sigma_1 += sigma_shape1[sigma_shape_off + k] *
                     sigma_vals[sigma_map[sigma_map_off_1 + k]];
          sigma_2 += sigma_shape2[sigma_shape_off + k] *
                     sigma_nbr;
        }
      }

      real_t speed1 = 0.0;
      real_t speed2 = 0.0;
      ComputeEulerFluxDotN_(u1, normal, dim, gamma, flux1, speed1);
      ComputeEulerFluxDotN_(u2, normal, dim, gamma, flux2, speed2);

      real_t cfl_speed1 = speed1;
      real_t cfl_speed2 = speed2;
      real_t dissipation_speed1 = speed1;
      real_t dissipation_speed2 = speed2;
      if (use_sigma)
      {
        AddSigmaDotNTerms_(
          u1, normal, dim, sigma_1, sigma_rho_floor, flux1);
        AddSigmaDotNTerms_(
          u2, normal, dim, sigma_2, sigma_rho_floor, flux2);
        const real_t igr_speed1 = IGRFluxCharacteristicSpeed(
          speed1, u1[0], sigma_1, sigma_rho_floor);
        const real_t igr_speed2 = IGRFluxCharacteristicSpeed(
          speed2, u2[0], sigma_2, sigma_rho_floor);
        cfl_speed1 = sigma_aware_cfl ? igr_speed1 : speed1;
        cfl_speed2 = sigma_aware_cfl ? igr_speed2 : speed2;
        dissipation_speed1 = sigma_aware_rusanov ? igr_speed1 : speed1;
        dissipation_speed2 = sigma_aware_rusanov ? igr_speed2 : speed2;
      }

      real_t normal_norm2 = 0.0;
      for (int d = 0; d < dim; ++d)
      {
        normal_norm2 += normal[d] * normal[d];
      }
      const real_t scaled_max_speed =
        fmax(dissipation_speed1, dissipation_speed2) * sqrt(normal_norm2);
      speed_max = fmax(speed_max, fmax(cfl_speed1, cfl_speed2));

      for (int eq = 0; eq < neq; ++eq)
      {
        flux_hat[eq] = 0.5 * (scaled_max_speed * (u1[eq] - u2[eq]) + flux1[eq] +
                              flux2[eq]);
      }

      const real_t wq = weights[face_q];
      for (int i = 0; i < dof; ++i)
      {
        const real_t c1 = -wq * shape1[shape_off + i];
        for (int eq = 0; eq < neq; ++eq)
        {
          const int zi1 = state_map[state_off_1 + i * neq + eq];
          AtomicAdd(z[zi1], c1 * flux_hat[eq]);
        }
      }
    }

    face_speed[f] = speed_max;
  }
};

struct PackedBoundaryFaceEulerRusanovKernel
{
  int dof = 0;
  int neq = 0;
  int dim = 0;
  int nq = 0;
  int ldof = 0;
  int sigma_ldof = 0;
  real_t gamma_default = 1.4;
  real_t sigma_rho_floor = 1.0e-12;
  bool use_sigma = false;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
  int bdr_max_attr = 0;
  const int* state_map = nullptr;
  const int* face_e1 = nullptr;
  const int* face_attr = nullptr;
  const real_t* shape1 = nullptr;
  const real_t* normals = nullptr;
  const real_t* weights = nullptr;
  const real_t* x = nullptr;
  real_t* z = nullptr;
  real_t* face_speed = nullptr;
  const int* bdr_type_by_attr = nullptr;
  const real_t* bdr_gamma_by_attr = nullptr;
  const real_t* bdr_pressure_by_attr = nullptr;
  const real_t* bdr_state_by_attr = nullptr;
  const int* sigma_map = nullptr;
  const real_t* sigma_vals = nullptr;
  const real_t* sigma_shape1 = nullptr;

  MFEM_HOST_DEVICE void operator()(int f) const
  {
    const int e1 = face_e1[f];
    const int state_off = e1 * ldof;
    const int attr = face_attr[f];
    if (attr <= 0 || attr > bdr_max_attr)
    {
      face_speed[f] = 0.0;
      return;
    }

    const int bdr_type = bdr_type_by_attr[attr];
    if (bdr_type == static_cast<int>(DeviceBoundaryType::Unsupported))
    {
      face_speed[f] = 0.0;
      return;
    }

    real_t gamma = bdr_gamma_by_attr[attr];
    if (!(gamma > 0.0))
    {
      gamma = gamma_default;
    }
    const real_t p_out = bdr_pressure_by_attr[attr];
    const real_t* state_ref = bdr_state_by_attr + attr * neq;

    const int sigma_map_off = e1 * sigma_ldof;

    real_t u1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t u2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux_hat[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t normal[3] = {0.0, 0.0, 0.0};

    real_t speed_max = 0.0;
    for (int q = 0; q < nq; ++q)
    {
      const int face_q = f * nq + q;
      const int shape_off = face_q * dof;
      const int normal_off = face_q * dim;

      for (int d = 0; d < dim; ++d)
      {
        normal[d] = normals[normal_off + d];
      }

      for (int eq = 0; eq < neq; ++eq)
      {
        real_t lhs = 0.0;
        for (int i = 0; i < dof; ++i)
        {
          const real_t s1 = shape1[shape_off + i];
          lhs += s1 * x[state_map[state_off + i * neq + eq]];
        }
        u1[eq] = lhs;
      }

      if (!EvalBoundaryGhostEuler_(bdr_type,
                                   dim,
                                   neq,
                                   gamma,
                                   p_out,
                                   state_ref,
                                   normal,
                                   u1,
                                   u2))
      {
        face_speed[f] = 0.0;
        return;
      }

      real_t sigma_1 = 0.0;
      if (use_sigma)
      {
        const int sigma_shape_off = face_q * sigma_ldof;
        for (int k = 0; k < sigma_ldof; ++k)
        {
          sigma_1 += sigma_shape1[sigma_shape_off + k] *
                     sigma_vals[sigma_map[sigma_map_off + k]];
        }
      }

      real_t speed1 = 0.0;
      real_t speed2 = 0.0;
      ComputeEulerFluxDotN_(u1, normal, dim, gamma, flux1, speed1);
      ComputeEulerFluxDotN_(u2, normal, dim, gamma, flux2, speed2);

      real_t cfl_speed1 = speed1;
      real_t cfl_speed2 = speed2;
      real_t dissipation_speed1 = speed1;
      real_t dissipation_speed2 = speed2;
      if (use_sigma)
      {
        AddSigmaDotNTerms_(
          u1, normal, dim, sigma_1, sigma_rho_floor, flux1);
        // For boundary faces FluxWithIGR side-explicit path samples left sigma
        // for both evaluations (Elem2 is null); mirror that behavior.
        AddSigmaDotNTerms_(
          u2, normal, dim, sigma_1, sigma_rho_floor, flux2);
        const real_t igr_speed1 = IGRFluxCharacteristicSpeed(
          speed1, u1[0], sigma_1, sigma_rho_floor);
        const real_t igr_speed2 = IGRFluxCharacteristicSpeed(
          speed2, u2[0], sigma_1, sigma_rho_floor);
        cfl_speed1 = sigma_aware_cfl ? igr_speed1 : speed1;
        cfl_speed2 = sigma_aware_cfl ? igr_speed2 : speed2;
        dissipation_speed1 = sigma_aware_rusanov ? igr_speed1 : speed1;
        dissipation_speed2 = sigma_aware_rusanov ? igr_speed2 : speed2;
      }

      real_t normal_norm2 = 0.0;
      for (int d = 0; d < dim; ++d)
      {
        normal_norm2 += normal[d] * normal[d];
      }
      const real_t scaled_max_speed =
        fmax(dissipation_speed1, dissipation_speed2) * sqrt(normal_norm2);
      speed_max = fmax(speed_max, fmax(cfl_speed1, cfl_speed2));

      for (int eq = 0; eq < neq; ++eq)
      {
        flux_hat[eq] = 0.5 * (scaled_max_speed * (u1[eq] - u2[eq]) + flux1[eq] +
                              flux2[eq]);
      }

      const real_t wq = weights[face_q];
      for (int i = 0; i < dof; ++i)
      {
        const real_t c1 = -wq * shape1[shape_off + i];
        for (int eq = 0; eq < neq; ++eq)
        {
          const int zi = state_map[state_off + i * neq + eq];
          AtomicAdd(z[zi], c1 * flux_hat[eq]);
        }
      }
    }
    face_speed[f] = speed_max;
  }
};

// Q-parallel variants: each thread handles one (face, q) sample and scatters
// contributions atomically. A separate reduction kernel computes per-face max
// characteristic speed.
struct PackedInteriorFaceEulerRusanovQKernel
{
  int dof = 0;
  int neq = 0;
  int dim = 0;
  int nq = 0;
  int ldof = 0;
  int sigma_ldof = 0;
  real_t gamma = 1.4;
  real_t sigma_rho_floor = 1.0e-12;
  bool use_sigma = false;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
  const int* state_map = nullptr;
  const int* face_e1 = nullptr;
  const int* face_e2 = nullptr;
  const real_t* shape1 = nullptr;
  const real_t* shape2 = nullptr;
  const real_t* normals = nullptr;
  const real_t* weights = nullptr;
  const real_t* x = nullptr;
  real_t* z = nullptr;
  real_t* face_speed_q = nullptr;
  const int* sigma_map = nullptr;
  const real_t* sigma_vals = nullptr;
  const real_t* sigma_shape1 = nullptr;
  const real_t* sigma_shape2 = nullptr;

  MFEM_HOST_DEVICE void operator()(int fq) const
  {
    const int f = fq / nq;
    const int q = fq - f * nq;
    const int e1 = face_e1[f];
    const int e2 = face_e2[f];
    const int state_off_1 = e1 * ldof;
    const int state_off_2 = e2 * ldof;
    const int face_q = f * nq + q;
    const int shape_off = face_q * dof;
    const int normal_off = face_q * dim;

    real_t u1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t u2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux_hat[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t normal[3] = {0.0, 0.0, 0.0};

    for (int d = 0; d < dim; ++d)
    {
      normal[d] = normals[normal_off + d];
    }

    for (int eq = 0; eq < neq; ++eq)
    {
      real_t lhs = 0.0;
      real_t rhs = 0.0;
      for (int i = 0; i < dof; ++i)
      {
        const real_t s1 = shape1[shape_off + i];
        const real_t s2 = shape2[shape_off + i];
        lhs += s1 * x[state_map[state_off_1 + i * neq + eq]];
        rhs += s2 * x[state_map[state_off_2 + i * neq + eq]];
      }
      u1[eq] = lhs;
      u2[eq] = rhs;
    }

    real_t sigma_1 = 0.0;
    real_t sigma_2 = 0.0;
    if (use_sigma)
    {
      const int sigma_shape_off = face_q * sigma_ldof;
      const int sigma_map_off_1 = e1 * sigma_ldof;
      const int sigma_map_off_2 = e2 * sigma_ldof;
      for (int k = 0; k < sigma_ldof; ++k)
      {
        sigma_1 += sigma_shape1[sigma_shape_off + k] *
                   sigma_vals[sigma_map[sigma_map_off_1 + k]];
        sigma_2 += sigma_shape2[sigma_shape_off + k] *
                   sigma_vals[sigma_map[sigma_map_off_2 + k]];
      }
    }

    real_t speed1 = 0.0;
    real_t speed2 = 0.0;
    ComputeEulerFluxDotN_(u1, normal, dim, gamma, flux1, speed1);
    ComputeEulerFluxDotN_(u2, normal, dim, gamma, flux2, speed2);

    real_t cfl_speed1 = speed1;
    real_t cfl_speed2 = speed2;
    real_t dissipation_speed1 = speed1;
    real_t dissipation_speed2 = speed2;
    if (use_sigma)
    {
      AddSigmaDotNTerms_(
        u1, normal, dim, sigma_1, sigma_rho_floor, flux1);
      AddSigmaDotNTerms_(
        u2, normal, dim, sigma_2, sigma_rho_floor, flux2);
      const real_t igr_speed1 = IGRFluxCharacteristicSpeed(
        speed1, u1[0], sigma_1, sigma_rho_floor);
      const real_t igr_speed2 = IGRFluxCharacteristicSpeed(
        speed2, u2[0], sigma_2, sigma_rho_floor);
      cfl_speed1 = sigma_aware_cfl ? igr_speed1 : speed1;
      cfl_speed2 = sigma_aware_cfl ? igr_speed2 : speed2;
      dissipation_speed1 = sigma_aware_rusanov ? igr_speed1 : speed1;
      dissipation_speed2 = sigma_aware_rusanov ? igr_speed2 : speed2;
    }

    real_t normal_norm2 = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      normal_norm2 += normal[d] * normal[d];
    }
    const real_t scaled_max_speed =
      fmax(dissipation_speed1, dissipation_speed2) * sqrt(normal_norm2);
    face_speed_q[fq] = fmax(cfl_speed1, cfl_speed2);

    for (int eq = 0; eq < neq; ++eq)
    {
      flux_hat[eq] = 0.5 * (scaled_max_speed * (u1[eq] - u2[eq]) + flux1[eq] +
                            flux2[eq]);
    }

    const real_t wq = weights[face_q];
    for (int i = 0; i < dof; ++i)
    {
      const real_t c1 = -wq * shape1[shape_off + i];
      const real_t c2 = +wq * shape2[shape_off + i];
      for (int eq = 0; eq < neq; ++eq)
      {
        const int zi1 = state_map[state_off_1 + i * neq + eq];
        const int zi2 = state_map[state_off_2 + i * neq + eq];
        AtomicAdd(z[zi1], c1 * flux_hat[eq]);
        AtomicAdd(z[zi2], c2 * flux_hat[eq]);
      }
    }
  }
};

struct PackedSharedFaceEulerRusanovQKernel
{
  int dof = 0;
  int neq = 0;
  int dim = 0;
  int nq = 0;
  int ldof = 0;
  int sigma_ldof = 0;
  real_t gamma = 1.4;
  real_t sigma_rho_floor = 1.0e-12;
  bool use_sigma = false;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
  const int* state_map = nullptr;
  const int* face_e1 = nullptr;
  const int* face_nbr_vdof_map = nullptr;
  const real_t* shape1 = nullptr;
  const real_t* shape2 = nullptr;
  const real_t* normals = nullptr;
  const real_t* weights = nullptr;
  const real_t* x = nullptr;
  const real_t* x_nbr = nullptr;
  real_t* z = nullptr;
  real_t* face_speed_q = nullptr;
  const int* sigma_map = nullptr;
  const real_t* sigma_vals = nullptr;
  const int* sigma_nbr_dof_map = nullptr;
  const real_t* sigma_nbr_vals = nullptr;
  const real_t* sigma_shape1 = nullptr;
  const real_t* sigma_shape2 = nullptr;

  MFEM_HOST_DEVICE void operator()(int fq) const
  {
    const int f = fq / nq;
    const int q = fq - f * nq;
    const int e1 = face_e1[f];
    const int state_off_1 = e1 * ldof;
    const int nbr_map_off = f * ldof;
    const int face_q = f * nq + q;
    const int shape_off = face_q * dof;
    const int normal_off = face_q * dim;

    real_t u1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t u2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux_hat[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t normal[3] = {0.0, 0.0, 0.0};

    for (int d = 0; d < dim; ++d)
    {
      normal[d] = normals[normal_off + d];
    }

    for (int eq = 0; eq < neq; ++eq)
    {
      real_t lhs = 0.0;
      real_t rhs = 0.0;
      for (int i = 0; i < dof; ++i)
      {
        const real_t s1 = shape1[shape_off + i];
        const real_t s2 = shape2[shape_off + i];
        const int vd_nbr = face_nbr_vdof_map[nbr_map_off + i * neq + eq];
        const real_t u2_i = (vd_nbr >= 0) ? x_nbr[vd_nbr] : -x_nbr[-1 - vd_nbr];
        lhs += s1 * x[state_map[state_off_1 + i * neq + eq]];
        rhs += s2 * u2_i;
      }
      u1[eq] = lhs;
      u2[eq] = rhs;
    }

    real_t sigma_1 = 0.0;
    real_t sigma_2 = 0.0;
    if (use_sigma)
    {
      const int sigma_shape_off = face_q * sigma_ldof;
      const int sigma_map_off_1 = e1 * sigma_ldof;
      const int sigma_map_off_2 = f * sigma_ldof;
      for (int k = 0; k < sigma_ldof; ++k)
      {
        const int sigma_nbr_d = sigma_nbr_dof_map[sigma_map_off_2 + k];
        const real_t sigma_nbr =
          (sigma_nbr_d >= 0) ? sigma_nbr_vals[sigma_nbr_d]
                             : -sigma_nbr_vals[-1 - sigma_nbr_d];
        sigma_1 += sigma_shape1[sigma_shape_off + k] *
                   sigma_vals[sigma_map[sigma_map_off_1 + k]];
        sigma_2 += sigma_shape2[sigma_shape_off + k] * sigma_nbr;
      }
    }

    real_t speed1 = 0.0;
    real_t speed2 = 0.0;
    ComputeEulerFluxDotN_(u1, normal, dim, gamma, flux1, speed1);
    ComputeEulerFluxDotN_(u2, normal, dim, gamma, flux2, speed2);

    real_t cfl_speed1 = speed1;
    real_t cfl_speed2 = speed2;
    real_t dissipation_speed1 = speed1;
    real_t dissipation_speed2 = speed2;
    if (use_sigma)
    {
      AddSigmaDotNTerms_(
        u1, normal, dim, sigma_1, sigma_rho_floor, flux1);
      AddSigmaDotNTerms_(
        u2, normal, dim, sigma_2, sigma_rho_floor, flux2);
      const real_t igr_speed1 = IGRFluxCharacteristicSpeed(
        speed1, u1[0], sigma_1, sigma_rho_floor);
      const real_t igr_speed2 = IGRFluxCharacteristicSpeed(
        speed2, u2[0], sigma_2, sigma_rho_floor);
      cfl_speed1 = sigma_aware_cfl ? igr_speed1 : speed1;
      cfl_speed2 = sigma_aware_cfl ? igr_speed2 : speed2;
      dissipation_speed1 = sigma_aware_rusanov ? igr_speed1 : speed1;
      dissipation_speed2 = sigma_aware_rusanov ? igr_speed2 : speed2;
    }

    real_t normal_norm2 = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      normal_norm2 += normal[d] * normal[d];
    }
    const real_t scaled_max_speed =
      fmax(dissipation_speed1, dissipation_speed2) * sqrt(normal_norm2);
    face_speed_q[fq] = fmax(cfl_speed1, cfl_speed2);

    for (int eq = 0; eq < neq; ++eq)
    {
      flux_hat[eq] = 0.5 * (scaled_max_speed * (u1[eq] - u2[eq]) + flux1[eq] +
                            flux2[eq]);
    }

    const real_t wq = weights[face_q];
    for (int i = 0; i < dof; ++i)
    {
      const real_t c1 = -wq * shape1[shape_off + i];
      for (int eq = 0; eq < neq; ++eq)
      {
        const int zi1 = state_map[state_off_1 + i * neq + eq];
        AtomicAdd(z[zi1], c1 * flux_hat[eq]);
      }
    }
  }
};

struct PackedBoundaryFaceEulerRusanovQKernel
{
  int dof = 0;
  int neq = 0;
  int dim = 0;
  int nq = 0;
  int ldof = 0;
  int sigma_ldof = 0;
  real_t gamma_default = 1.4;
  real_t sigma_rho_floor = 1.0e-12;
  bool use_sigma = false;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
  int bdr_max_attr = 0;
  const int* state_map = nullptr;
  const int* face_e1 = nullptr;
  const int* face_attr = nullptr;
  const real_t* shape1 = nullptr;
  const real_t* normals = nullptr;
  const real_t* weights = nullptr;
  const real_t* x = nullptr;
  real_t* z = nullptr;
  real_t* face_speed_q = nullptr;
  const int* bdr_type_by_attr = nullptr;
  const real_t* bdr_gamma_by_attr = nullptr;
  const real_t* bdr_pressure_by_attr = nullptr;
  const real_t* bdr_state_by_attr = nullptr;
  const int* sigma_map = nullptr;
  const real_t* sigma_vals = nullptr;
  const real_t* sigma_shape1 = nullptr;

  MFEM_HOST_DEVICE void operator()(int fq) const
  {
    const int f = fq / nq;
    const int q = fq - f * nq;
    const int e1 = face_e1[f];
    const int state_off = e1 * ldof;
    const int attr = face_attr[f];
    if (attr <= 0 || attr > bdr_max_attr)
    {
      face_speed_q[fq] = 0.0;
      return;
    }

    const int bdr_type = bdr_type_by_attr[attr];
    if (bdr_type == static_cast<int>(DeviceBoundaryType::Unsupported))
    {
      face_speed_q[fq] = 0.0;
      return;
    }

    real_t gamma = bdr_gamma_by_attr[attr];
    if (!(gamma > 0.0))
    {
      gamma = gamma_default;
    }
    const real_t p_out = bdr_pressure_by_attr[attr];
    const real_t* state_ref = bdr_state_by_attr + attr * neq;

    const int face_q = f * nq + q;
    const int shape_off = face_q * dof;
    const int normal_off = face_q * dim;
    const int sigma_map_off = e1 * sigma_ldof;

    real_t u1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t u2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux2[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t flux_hat[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    real_t normal[3] = {0.0, 0.0, 0.0};

    for (int d = 0; d < dim; ++d)
    {
      normal[d] = normals[normal_off + d];
    }

    for (int eq = 0; eq < neq; ++eq)
    {
      real_t lhs = 0.0;
      for (int i = 0; i < dof; ++i)
      {
        const real_t s1 = shape1[shape_off + i];
        lhs += s1 * x[state_map[state_off + i * neq + eq]];
      }
      u1[eq] = lhs;
    }

    if (!EvalBoundaryGhostEuler_(bdr_type,
                                 dim,
                                 neq,
                                 gamma,
                                 p_out,
                                 state_ref,
                                 normal,
                                 u1,
                                 u2))
    {
      face_speed_q[fq] = 0.0;
      return;
    }

    real_t sigma_1 = 0.0;
    if (use_sigma)
    {
      const int sigma_shape_off = face_q * sigma_ldof;
      for (int k = 0; k < sigma_ldof; ++k)
      {
        sigma_1 += sigma_shape1[sigma_shape_off + k] *
                   sigma_vals[sigma_map[sigma_map_off + k]];
      }
    }

    real_t speed1 = 0.0;
    real_t speed2 = 0.0;
    ComputeEulerFluxDotN_(u1, normal, dim, gamma, flux1, speed1);
    ComputeEulerFluxDotN_(u2, normal, dim, gamma, flux2, speed2);

    real_t cfl_speed1 = speed1;
    real_t cfl_speed2 = speed2;
    real_t dissipation_speed1 = speed1;
    real_t dissipation_speed2 = speed2;
    if (use_sigma)
    {
      AddSigmaDotNTerms_(
        u1, normal, dim, sigma_1, sigma_rho_floor, flux1);
      AddSigmaDotNTerms_(
        u2, normal, dim, sigma_1, sigma_rho_floor, flux2);
      const real_t igr_speed1 = IGRFluxCharacteristicSpeed(
        speed1, u1[0], sigma_1, sigma_rho_floor);
      const real_t igr_speed2 = IGRFluxCharacteristicSpeed(
        speed2, u2[0], sigma_1, sigma_rho_floor);
      cfl_speed1 = sigma_aware_cfl ? igr_speed1 : speed1;
      cfl_speed2 = sigma_aware_cfl ? igr_speed2 : speed2;
      dissipation_speed1 = sigma_aware_rusanov ? igr_speed1 : speed1;
      dissipation_speed2 = sigma_aware_rusanov ? igr_speed2 : speed2;
    }

    real_t normal_norm2 = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      normal_norm2 += normal[d] * normal[d];
    }
    const real_t scaled_max_speed =
      fmax(dissipation_speed1, dissipation_speed2) * sqrt(normal_norm2);
    face_speed_q[fq] = fmax(cfl_speed1, cfl_speed2);

    for (int eq = 0; eq < neq; ++eq)
    {
      flux_hat[eq] = 0.5 * (scaled_max_speed * (u1[eq] - u2[eq]) + flux1[eq] +
                            flux2[eq]);
    }

    const real_t wq = weights[face_q];
    for (int i = 0; i < dof; ++i)
    {
      const real_t c1 = -wq * shape1[shape_off + i];
      for (int eq = 0; eq < neq; ++eq)
      {
        const int zi = state_map[state_off + i * neq + eq];
        AtomicAdd(z[zi], c1 * flux_hat[eq]);
      }
    }
  }
};

struct FaceSpeedQReduceKernel
{
  int nq = 0;
  const real_t* face_speed_q = nullptr;
  real_t* face_speed = nullptr;

  MFEM_HOST_DEVICE void operator()(int f) const
  {
    const int off = f * nq;
    real_t max_speed = 0.0;
    for (int q = 0; q < nq; ++q)
    {
      max_speed = fmax(max_speed, face_speed_q[off + q]);
    }
    face_speed[f] = max_speed;
  }
};

struct MaxSpeedReductionBody
{
  const real_t* speed = nullptr;

  MFEM_HOST_DEVICE void operator()(int i, real_t& local_max) const
  {
    local_max = fmax(local_max, speed[i]);
  }
};

#endif

} // namespace

// Internal-only helper/kernels for DGHyperbolicConservationLaws device path.
// This file is intentionally included from
// `DGHyperbolicConservationLaws.cpp` after MFEM/CUDA/HIP backend headers.
#pragma once

#include "igrlab/compat.hpp"

namespace
{

using DeviceBoundaryType = detail::EulerBoundaryStateKind;

constexpr int kFrozenSigmaTelemetryWidth_ = 5;
constexpr int kFrozenSigmaSampleCount_ = 0;
constexpr int kFrozenSigmaNonpositiveCount_ = 1;
constexpr int kFrozenSigmaNonfiniteCount_ = 2;
constexpr int kFrozenSigmaMinimumRaw_ = 3;
constexpr int kFrozenSigmaMinimumRatio_ = 4;

#ifdef DG_IGR_HAS_ACCEL_FORALL
MFEM_HOST_DEVICE inline bool FiniteTelemetryValue_(const double value)
{
  return value == value &&
         fabs(value) <= std::numeric_limits<double>::max();
}

/** Accumulate one actual invariant-speed sample into a uniquely owned slot.
 *
 * Packed kernels assign exactly one thread to each slot, so no atomics are
 * required.  The slot persists across Mult() calls and is read only when the
 * application requests the final bounded aggregate.
 */
MFEM_HOST_DEVICE inline void AccumulateFrozenSigmaTelemetry_(
  const real_t pressure,
  const real_t sigma,
  const real_t gamma,
  double* const slot)
{
  if (!slot)
  {
    return;
  }
  const double gamma64 = static_cast<double>(gamma);
  const double pressure64 = static_cast<double>(pressure);
  const double sigma64 = static_cast<double>(sigma);
  const double q_sigma =
    gamma64 * pressure64 + (gamma64 - 1.0) * sigma64;
  const double ratio = q_sigma / (gamma64 * pressure64);
  slot[kFrozenSigmaSampleCount_] += 1.0;
  const bool q_finite = FiniteTelemetryValue_(q_sigma);
  const bool ratio_finite = FiniteTelemetryValue_(ratio);
  if (!q_finite || !ratio_finite)
  {
    slot[kFrozenSigmaNonfiniteCount_] += 1.0;
  }
  if (q_finite)
  {
    if (q_sigma <= 0.0)
    {
      slot[kFrozenSigmaNonpositiveCount_] += 1.0;
    }
    slot[kFrozenSigmaMinimumRaw_] =
      fmin(slot[kFrozenSigmaMinimumRaw_], q_sigma);
  }
  if (ratio_finite)
  {
    slot[kFrozenSigmaMinimumRatio_] =
      fmin(slot[kFrozenSigmaMinimumRatio_], ratio);
  }
}

MFEM_HOST_DEVICE inline bool EvalBoundaryGhostEuler_(
  const int boundary_type,
  const int dimension,
  const int equations,
  const real_t gamma,
  const real_t pressure_out,
  const real_t* reference_state,
  const real_t* normal,
  const real_t* interior,
  real_t* exterior,
  const bool invariant_domain_flux,
  const real_t invariant_domain_density_floor,
  const real_t invariant_domain_pressure_floor)
{
  if (invariant_domain_flux)
  {
    if (!detail::EulerConservativeStateMeetsContract(
          interior,
          dimension,
          equations,
          gamma,
          invariant_domain_density_floor,
          invariant_domain_pressure_floor))
      return false;
    const bool evaluated = detail::EvaluateEulerBoundaryState(
      static_cast<DeviceBoundaryType>(boundary_type),
      dimension,
      equations,
      gamma,
      pressure_out,
      reference_state,
      normal,
      interior,
      exterior,
      invariant_domain_density_floor,
      invariant_domain_pressure_floor);
    return evaluated && detail::EulerConservativeStateMeetsContract(
      exterior,
      dimension,
      equations,
      gamma,
      invariant_domain_density_floor,
      invariant_domain_pressure_floor);
  }
  return detail::EvaluateEulerBoundaryState(
      static_cast<DeviceBoundaryType>(boundary_type),
      dimension,
      equations,
      gamma,
      pressure_out,
      reference_state,
      normal,
      interior,
      exterior);
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
  ::igrlab::frozen_scalar_igr_algebra::AddNormalSigmaFlux(dim,
                                                          state,
                                                          1,
                                                          normal,
                                                          1,
                                                          sigma,
                                                          rho_floor,
                                                          flux_dot_n,
                                                          1);
}

MFEM_HOST_DEVICE inline real_t IGRInvariantDomainFaceSpeed_(
  const real_t* state,
  const real_t* normal,
  int dim,
  real_t gamma,
  real_t base_speed,
  real_t sigma,
  real_t rho_floor,
  real_t pressure_floor,
  double* const telemetry_slot)
{
  const real_t density = state[0];
  const real_t safe_density = fmax(density, rho_floor);
  real_t momentum2 = 0.0;
  real_t normal_momentum = 0.0;
  real_t normal_norm2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const real_t momentum_d = state[1 + d];
    momentum2 += momentum_d * momentum_d;
    normal_momentum += momentum_d * normal[d];
    normal_norm2 += normal[d] * normal[d];
  }
  const real_t pressure =
    (gamma - 1.0) * (state[1 + dim] - 0.5 * momentum2 / safe_density);
  AccumulateFrozenSigmaTelemetry_(
    pressure, sigma, gamma, telemetry_slot);
  const real_t normal_norm = sqrt(normal_norm2);
  const real_t transport_speed =
    normal_norm > 0.0
      ? fabs(normal_momentum) / (safe_density * normal_norm)
      : 0.0;
  return IGRFluxInvariantDomainSpeed(base_speed,
                                     density,
                                     pressure,
                                     transport_speed,
                                     sigma,
                                     gamma,
                                     rho_floor,
                                     pressure_floor);
}

MFEM_HOST_DEVICE inline void SelectIGRFaceSpeeds_(
  const real_t* state,
  const real_t* normal,
  int dim,
  real_t gamma,
  real_t base_speed,
  real_t sigma,
  real_t rho_floor,
  real_t pressure_floor,
  bool invariant_domain_flux,
  bool sigma_aware_cfl,
  bool sigma_aware_rusanov,
  double* const q_sigma_telemetry_slot,
  real_t& cfl_speed,
  real_t& dissipation_speed)
{
  if (invariant_domain_flux)
  {
    const real_t bound = IGRInvariantDomainFaceSpeed_(state,
                                                      normal,
                                                      dim,
                                                      gamma,
                                                      base_speed,
                                                      sigma,
                                                      rho_floor,
                                                      pressure_floor,
                                                      q_sigma_telemetry_slot);
    cfl_speed = bound;
    dissipation_speed = bound;
    return;
  }

  const real_t heuristic_speed =
    IGRFluxCharacteristicSpeed(base_speed, state[0], sigma, rho_floor);
  cfl_speed = sigma_aware_cfl ? heuristic_speed : base_speed;
  dissipation_speed = sigma_aware_rusanov ? heuristic_speed : base_speed;
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
  real_t invariant_domain_pressure_floor = 1.0e-12;
  bool use_sigma = false;
  bool invariant_domain_flux = false;
  bool track_volume_speed = false;
  const int* state_map = nullptr;
  const real_t* weakdiv = nullptr;
  const real_t* x = nullptr;
  real_t* z = nullptr;
  real_t* volume_speed = nullptr;
  double* q_sigma_telemetry = nullptr;
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
    real_t element_max_speed = 0.0;

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
      if (use_sigma && invariant_domain_flux)
      {
        AccumulateFrozenSigmaTelemetry_(
          pressure,
          sigma_j,
          gamma,
          q_sigma_telemetry +
            (e * dof + j) * kFrozenSigmaTelemetryWidth_);
      }
      const real_t inv_rho_sigma =
        (use_sigma ? 1.0 / fmax(rho, sigma_rho_floor) : 0.0);
      if (track_volume_speed)
      {
        const real_t transport_speed = sqrt(mom2) * inv_rho;
        const real_t base_speed =
          transport_speed + sqrt(gamma * pressure * inv_rho);
        const real_t nodal_speed =
          IGRFluxInvariantDomainSpeed(base_speed,
                                      rho,
                                      pressure,
                                      transport_speed,
                                      sigma_j,
                                      gamma,
                                      sigma_rho_floor,
                                      invariant_domain_pressure_floor);
        element_max_speed = fmax(element_max_speed, nodal_speed);
      }

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
                flux_md += ::igrlab::frozen_scalar_igr_algebra::
                  TensorSigmaMomentumDiagonalTerm(sigma_j);
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
            flux_ed += ::igrlab::frozen_scalar_igr_algebra::
              TensorSigmaEnergyTermFromInverseDensity(
                sigma_j, mom[d], inv_rho_sigma);
          }
          sum_e += weakdiv[wbase + d] * flux_ed;
        }
        z[state_map[state_off + i * neq + energy_eq]] += sum_e;
      }
    }
    if (track_volume_speed)
    {
      volume_speed[e] = element_max_speed;
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
  real_t invariant_domain_pressure_floor = 1.0e-12;
  bool use_sigma = false;
  bool invariant_domain_flux = false;
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
  const real_t* reference_x_slopes = nullptr;
  const real_t* ip1_ref = nullptr;
  const real_t* ip2_ref = nullptr;
  real_t* z = nullptr;
  real_t* face_speed = nullptr;
  double* q_sigma_telemetry = nullptr;
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
        if (reference_x_slopes)
        {
          lhs += (ip1_ref[normal_off]-.5) * reference_x_slopes[state_map[state_off_1+eq]];
          rhs += (ip2_ref[normal_off]-.5) * reference_x_slopes[state_map[state_off_2+eq]];
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
        double* const telemetry_slot =
          invariant_domain_flux
            ? q_sigma_telemetry + face_q * kFrozenSigmaTelemetryWidth_
            : nullptr;
        SelectIGRFaceSpeeds_(u1,
                             normal,
                             dim,
                             gamma,
                             speed1,
                             sigma_1,
                             sigma_rho_floor,
                             invariant_domain_pressure_floor,
                             invariant_domain_flux,
                             sigma_aware_cfl,
                             sigma_aware_rusanov,
                             telemetry_slot,
                             cfl_speed1,
                             dissipation_speed1);
        SelectIGRFaceSpeeds_(u2,
                             normal,
                             dim,
                             gamma,
                             speed2,
                             sigma_2,
                             sigma_rho_floor,
                             invariant_domain_pressure_floor,
                             invariant_domain_flux,
                             sigma_aware_cfl,
                             sigma_aware_rusanov,
                             telemetry_slot,
                             cfl_speed2,
                             dissipation_speed2);
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
  real_t invariant_domain_pressure_floor = 1.0e-12;
  bool use_sigma = false;
  bool invariant_domain_flux = false;
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
  double* q_sigma_telemetry = nullptr;
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
        double* const telemetry_slot =
          invariant_domain_flux
            ? q_sigma_telemetry + face_q * kFrozenSigmaTelemetryWidth_
            : nullptr;
        SelectIGRFaceSpeeds_(u1,
                             normal,
                             dim,
                             gamma,
                             speed1,
                             sigma_1,
                             sigma_rho_floor,
                             invariant_domain_pressure_floor,
                             invariant_domain_flux,
                             sigma_aware_cfl,
                             sigma_aware_rusanov,
                             telemetry_slot,
                             cfl_speed1,
                             dissipation_speed1);
        SelectIGRFaceSpeeds_(u2,
                             normal,
                             dim,
                             gamma,
                             speed2,
                             sigma_2,
                             sigma_rho_floor,
                             invariant_domain_pressure_floor,
                             invariant_domain_flux,
                             sigma_aware_cfl,
                             sigma_aware_rusanov,
                             telemetry_slot,
                             cfl_speed2,
                             dissipation_speed2);
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
  real_t invariant_domain_pressure_floor = 1.0e-12;
  bool use_sigma = false;
  bool invariant_domain_flux = false;
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
  const real_t* reference_x_slopes = nullptr;
  const real_t* ip1_ref = nullptr;
  real_t* z = nullptr;
  real_t* face_speed = nullptr;
  double* q_sigma_telemetry = nullptr;
  real_t* boundary_flux_integral = nullptr;
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
    if (bdr_type == static_cast<int>(DeviceBoundaryType::unsupported))
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
        if (reference_x_slopes)
          lhs += (ip1_ref[normal_off]-.5) * reference_x_slopes[state_map[state_off+eq]];
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
                                   u2,
                                   invariant_domain_flux,
                                   sigma_rho_floor,
                                   invariant_domain_pressure_floor))
      {
        const real_t infinity = std::numeric_limits<real_t>::infinity();
        const real_t invalid = std::numeric_limits<real_t>::quiet_NaN();
        face_speed[f] = infinity;
        if (boundary_flux_integral)
        {
          for (int eq = 0; eq < neq; ++eq)
          {
            AtomicAdd(boundary_flux_integral[eq], invalid);
          }
        }
        for (int i = 0; i < dof; ++i)
        {
          for (int eq = 0; eq < neq; ++eq)
          {
            const int zi = state_map[state_off + i * neq + eq];
            AtomicAdd(z[zi], invalid);
          }
        }
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
        double* const telemetry_slot =
          invariant_domain_flux
            ? q_sigma_telemetry + face_q * kFrozenSigmaTelemetryWidth_
            : nullptr;
        SelectIGRFaceSpeeds_(u1,
                             normal,
                             dim,
                             gamma,
                             speed1,
                             sigma_1,
                             sigma_rho_floor,
                             invariant_domain_pressure_floor,
                             invariant_domain_flux,
                             sigma_aware_cfl,
                             sigma_aware_rusanov,
                             telemetry_slot,
                             cfl_speed1,
                             dissipation_speed1);
        SelectIGRFaceSpeeds_(u2,
                             normal,
                             dim,
                             gamma,
                             speed2,
                             sigma_1,
                             sigma_rho_floor,
                             invariant_domain_pressure_floor,
                             invariant_domain_flux,
                             sigma_aware_cfl,
                             sigma_aware_rusanov,
                             telemetry_slot,
                             cfl_speed2,
                             dissipation_speed2);
      }

      real_t normal_norm2 = 0.0;
      for (int d = 0; d < dim; ++d)
      {
        normal_norm2 += normal[d] * normal[d];
      }
      const real_t scaled_max_speed =
        fmax(dissipation_speed1, dissipation_speed2) * sqrt(normal_norm2);
      speed_max = fmax(speed_max, fmax(cfl_speed1, cfl_speed2));
      const real_t wq = weights[face_q];

      for (int eq = 0; eq < neq; ++eq)
      {
        flux_hat[eq] = 0.5 * (scaled_max_speed * (u1[eq] - u2[eq]) + flux1[eq] +
                              flux2[eq]);
        // Record the outward flux before the residual's leading minus sign.
        if (boundary_flux_integral)
        {
          AtomicAdd(boundary_flux_integral[eq], wq * flux_hat[eq]);
        }
      }

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
  real_t invariant_domain_pressure_floor = 1.0e-12;
  bool use_sigma = false;
  bool invariant_domain_flux = false;
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
  const real_t* reference_x_slopes = nullptr;
  const real_t* ip1_ref = nullptr;
  const real_t* ip2_ref = nullptr;
  real_t* z = nullptr;
  real_t* face_speed_q = nullptr;
  double* q_sigma_telemetry = nullptr;
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
      if (reference_x_slopes)
      {
        lhs += (ip1_ref[normal_off]-.5) * reference_x_slopes[state_map[state_off_1+eq]];
        rhs += (ip2_ref[normal_off]-.5) * reference_x_slopes[state_map[state_off_2+eq]];
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
      double* const telemetry_slot =
        invariant_domain_flux
          ? q_sigma_telemetry + face_q * kFrozenSigmaTelemetryWidth_
          : nullptr;
      SelectIGRFaceSpeeds_(u1,
                           normal,
                           dim,
                           gamma,
                           speed1,
                           sigma_1,
                           sigma_rho_floor,
                           invariant_domain_pressure_floor,
                           invariant_domain_flux,
                           sigma_aware_cfl,
                           sigma_aware_rusanov,
                           telemetry_slot,
                           cfl_speed1,
                           dissipation_speed1);
      SelectIGRFaceSpeeds_(u2,
                           normal,
                           dim,
                           gamma,
                           speed2,
                           sigma_2,
                           sigma_rho_floor,
                           invariant_domain_pressure_floor,
                           invariant_domain_flux,
                           sigma_aware_cfl,
                           sigma_aware_rusanov,
                           telemetry_slot,
                           cfl_speed2,
                           dissipation_speed2);
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
  real_t invariant_domain_pressure_floor = 1.0e-12;
  bool use_sigma = false;
  bool invariant_domain_flux = false;
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
  double* q_sigma_telemetry = nullptr;
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
      double* const telemetry_slot =
        invariant_domain_flux
          ? q_sigma_telemetry + face_q * kFrozenSigmaTelemetryWidth_
          : nullptr;
      SelectIGRFaceSpeeds_(u1,
                           normal,
                           dim,
                           gamma,
                           speed1,
                           sigma_1,
                           sigma_rho_floor,
                           invariant_domain_pressure_floor,
                           invariant_domain_flux,
                           sigma_aware_cfl,
                           sigma_aware_rusanov,
                           telemetry_slot,
                           cfl_speed1,
                           dissipation_speed1);
      SelectIGRFaceSpeeds_(u2,
                           normal,
                           dim,
                           gamma,
                           speed2,
                           sigma_2,
                           sigma_rho_floor,
                           invariant_domain_pressure_floor,
                           invariant_domain_flux,
                           sigma_aware_cfl,
                           sigma_aware_rusanov,
                           telemetry_slot,
                           cfl_speed2,
                           dissipation_speed2);
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
  real_t invariant_domain_pressure_floor = 1.0e-12;
  bool use_sigma = false;
  bool invariant_domain_flux = false;
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
  const real_t* reference_x_slopes = nullptr;
  const real_t* ip1_ref = nullptr;
  real_t* z = nullptr;
  real_t* face_speed_q = nullptr;
  double* q_sigma_telemetry = nullptr;
  real_t* boundary_flux_integral = nullptr;
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
    if (bdr_type == static_cast<int>(DeviceBoundaryType::unsupported))
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
      if (reference_x_slopes)
        lhs += (ip1_ref[normal_off]-.5) * reference_x_slopes[state_map[state_off+eq]];
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
                                 u2,
                                 invariant_domain_flux,
                                 sigma_rho_floor,
                                 invariant_domain_pressure_floor))
    {
      const real_t infinity = std::numeric_limits<real_t>::infinity();
      const real_t invalid = std::numeric_limits<real_t>::quiet_NaN();
      face_speed_q[fq] = infinity;
      if (boundary_flux_integral)
      {
        for (int eq = 0; eq < neq; ++eq)
          AtomicAdd(boundary_flux_integral[eq], invalid);
      }
      for (int i = 0; i < dof; ++i)
      {
        for (int eq = 0; eq < neq; ++eq)
        {
          const int zi = state_map[state_off + i * neq + eq];
          AtomicAdd(z[zi], invalid);
        }
      }
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
      double* const telemetry_slot =
        invariant_domain_flux
          ? q_sigma_telemetry + face_q * kFrozenSigmaTelemetryWidth_
          : nullptr;
      SelectIGRFaceSpeeds_(u1,
                           normal,
                           dim,
                           gamma,
                           speed1,
                           sigma_1,
                           sigma_rho_floor,
                           invariant_domain_pressure_floor,
                           invariant_domain_flux,
                           sigma_aware_cfl,
                           sigma_aware_rusanov,
                           telemetry_slot,
                           cfl_speed1,
                           dissipation_speed1);
      SelectIGRFaceSpeeds_(u2,
                           normal,
                           dim,
                           gamma,
                           speed2,
                           sigma_1,
                           sigma_rho_floor,
                           invariant_domain_pressure_floor,
                           invariant_domain_flux,
                           sigma_aware_cfl,
                           sigma_aware_rusanov,
                           telemetry_slot,
                           cfl_speed2,
                           dissipation_speed2);
    }

    real_t normal_norm2 = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      normal_norm2 += normal[d] * normal[d];
    }
    const real_t scaled_max_speed =
      fmax(dissipation_speed1, dissipation_speed2) * sqrt(normal_norm2);
    face_speed_q[fq] = fmax(cfl_speed1, cfl_speed2);
    const real_t wq = weights[face_q];

    for (int eq = 0; eq < neq; ++eq)
    {
      flux_hat[eq] = 0.5 * (scaled_max_speed * (u1[eq] - u2[eq]) + flux1[eq] +
                            flux2[eq]);
      // Record the outward flux before the residual's leading minus sign.
      if (boundary_flux_integral)
      {
        AtomicAdd(boundary_flux_integral[eq], wq * flux_hat[eq]);
      }
    }

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

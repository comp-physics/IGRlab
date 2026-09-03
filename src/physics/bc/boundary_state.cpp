#include "physics/bc/boundary_state.hpp"

#include <cmath>

namespace mfem
{

namespace
{
struct PrimitiveNormalState
{
  double rho = 1.0;
  double p = 1.0;
  double un = 0.0;
  double ut = 0.0;
  double c = 1.0;
};

void BuildNormalTangential(const mfem::Vector& nor,
                           int dim,
                           mfem::Vector& nhat,
                           mfem::Vector& that)
{
  MFEM_VERIFY(dim == 1 || dim == 2,
              "Boundary-state helpers currently support 1D/2D Euler.");
  MFEM_VERIFY(nor.Size() >= dim,
              "Boundary normal has insufficient size for the requested dimension.");

  nhat.SetSize(dim);
  nhat = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    nhat(d) = nor(d);
  }
  const double nrm = nhat.Norml2();
  if (nrm > 0.0)
  {
    nhat /= nrm;
  }
  else
  {
    nhat = 0.0;
    nhat(0) = 1.0;
  }

  that.SetSize(dim);
  that = 0.0;
  if (dim == 2)
  {
    that(0) = -nhat(1);
    that(1) = nhat(0);
  }
}

PrimitiveNormalState ConservativeToPrimitiveNormal(const mfem::Vector& U,
                                                   int dim,
                                                   double gamma,
                                                   const mfem::Vector& nhat,
                                                   const mfem::Vector& that,
                                                   double density_floor,
                                                   double pressure_floor)
{
  PrimitiveNormalState out;
  MFEM_VERIFY(U.Size() >= dim + 2,
              "Conservative state has insufficient size for Euler variables.");

  const double rho = std::fmax(U(0), density_floor);
  mfem::Vector vel(dim);
  double vel2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    vel(d) = U(1 + d) / rho;
    vel2 += vel(d) * vel(d);
  }

  const double E = U(1 + dim) / rho;
  const double kinetic = 0.5 * vel2;
  const double p =
    std::fmax((gamma - 1.0) * rho * (E - kinetic), pressure_floor);
  const double c = std::sqrt(std::fmax(gamma * p / rho, 0.0));

  out.rho = rho;
  out.p = p;
  out.c = c;
  out.un = 0.0;
  out.ut = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    out.un += vel(d) * nhat(d);
  }
  if (dim == 2)
  {
    out.ut = vel(0) * that(0) + vel(1) * that(1);
  }

  return out;
}

void PrimitiveNormalToConservative(const PrimitiveNormalState& prim,
                                   int dim,
                                   double gamma,
                                   const mfem::Vector& nhat,
                                   const mfem::Vector& that,
                                   mfem::Vector& U)
{
  U.SetSize(dim + 2);

  mfem::Vector vel(dim);
  if (dim == 1)
  {
    vel(0) = prim.un;
  }
  else
  {
    vel(0) = prim.un * nhat(0) + prim.ut * that(0);
    vel(1) = prim.un * nhat(1) + prim.ut * that(1);
  }

  U(0) = prim.rho;
  double vel2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    U(1 + d) = prim.rho * vel(d);
    vel2 += vel(d) * vel(d);
  }
  const double E = prim.p / ((gamma - 1.0) * prim.rho) + 0.5 * vel2;
  U(1 + dim) = prim.rho * E;
}

inline double
EntropyLike(double p, double rho, double gamma, double density_floor)
{
  return p / std::pow(std::fmax(rho, density_floor), gamma);
}

PrimitiveNormalState SubsonicOutletPressureClosure(const PrimitiveNormalState& inner,
                                                   double gamma,
                                                   double p_out,
                                                   double density_floor,
                                                   double pressure_floor)
{
  PrimitiveNormalState out = inner;
  if (inner.un >= inner.c || inner.un <= -inner.c)
  {
    return out; // supersonic or reversed branch -> transmissive fallback
  }

  const double g1 = gamma - 1.0;
  const double s = EntropyLike(inner.p, inner.rho, gamma, density_floor);
  const double p = std::fmax(p_out, pressure_floor);
  const double rho = std::pow(p / s, 1.0 / gamma);
  const double c = std::sqrt(std::fmax(gamma * p / rho, 0.0));
  const double j_plus = inner.un + 2.0 * inner.c / g1;
  const double un = j_plus - 2.0 * c / g1;

  out.rho = std::fmax(rho, density_floor);
  out.p = p;
  out.c = c;
  out.un = un;
  out.ut = inner.ut;
  return out;
}

PrimitiveNormalState NonReflectingOutletClosure(const PrimitiveNormalState& inner,
                                                const PrimitiveNormalState& ref,
                                                double gamma,
                                                double density_floor,
                                                double pressure_floor)
{
  PrimitiveNormalState out = inner;
  if (inner.un >= inner.c || inner.un <= -inner.c)
  {
    return out; // supersonic or reversed branch -> transmissive fallback
  }

  const double g1 = gamma - 1.0;
  const double j_plus = inner.un + 2.0 * inner.c / g1;
  const double j_minus_ref = ref.un - 2.0 * ref.c / g1;
  const double un = 0.5 * (j_plus + j_minus_ref);
  const double c = 0.25 * g1 * (j_plus - j_minus_ref);
  if (!(c > 0.0))
  {
    return out;
  }

  const double s_ref = EntropyLike(ref.p, ref.rho, gamma, density_floor);
  const double entropy_scale = gamma * s_ref;
  if (!std::isfinite(s_ref) || !(s_ref > 0.0) ||
      !std::isfinite(entropy_scale) || !(entropy_scale > 0.0))
  {
    return out;
  }
  const double rho = std::pow((c * c) / entropy_scale, 1.0 / g1);
  if (!std::isfinite(rho) || !(rho > 0.0))
  {
    return out;
  }
  const double admitted_rho = std::fmax(rho, density_floor);
  const double p = s_ref * std::pow(admitted_rho, gamma);
  if (!std::isfinite(p) || p < 0.0)
  {
    return out;
  }

  out.rho = admitted_rho;
  out.p = std::fmax(p, pressure_floor);
  out.c = c;
  out.un = un;
  out.ut = inner.ut;
  return out;
}

PrimitiveNormalState SubsonicInletClosure(const PrimitiveNormalState& inner,
                                          const PrimitiveNormalState& target,
                                          double gamma,
                                          double density_floor,
                                          double pressure_floor)
{
  PrimitiveNormalState out = target;
  if (inner.un <= -inner.c)
  {
    return out; // supersonic inflow -> full target state
  }
  if (inner.un >= inner.c)
  {
    return inner; // unexpected outflow branch at inlet -> transmissive fallback
  }

  const double g1 = gamma - 1.0;
  const double j_plus_inner = inner.un + 2.0 * inner.c / g1;
  const double j_minus_target = target.un - 2.0 * target.c / g1;
  const double un = 0.5 * (j_plus_inner + j_minus_target);
  const double c = 0.25 * g1 * (j_plus_inner - j_minus_target);
  if (!(c > 0.0))
  {
    return target;
  }

  const double s_in = EntropyLike(target.p, target.rho, gamma, density_floor);
  const double entropy_scale = gamma * s_in;
  if (!std::isfinite(s_in) || !(s_in > 0.0) ||
      !std::isfinite(entropy_scale) || !(entropy_scale > 0.0))
  {
    return target;
  }
  const double rho = std::pow((c * c) / entropy_scale, 1.0 / g1);
  if (!std::isfinite(rho) || !(rho > 0.0))
  {
    return target;
  }
  const double admitted_rho = std::fmax(rho, density_floor);
  const double p = s_in * std::pow(admitted_rho, gamma);
  if (!std::isfinite(p) || p < 0.0)
  {
    return target;
  }

  out.rho = admitted_rho;
  out.p = std::fmax(p, pressure_floor);
  out.c = c;
  out.un = un;
  out.ut = target.ut;
  return out;
}

} // namespace

void EvaluateEulerExteriorState(const EulerExteriorStateKind kind,
                                const Vector& Uminus,
                                const Vector& nor,
                                const int dim,
                                const double gamma,
                                const Vector* const reference_state,
                                const double pressure_out,
                                Vector& Uplus,
                                const double density_floor,
                                const double pressure_floor)
{
  switch (kind)
  {
  case EulerExteriorStateKind::SupersonicInlet:
    MFEM_VERIFY(reference_state != nullptr,
                "Supersonic inlet exterior state requires a reference state.");
    Uplus = *reference_state;
    return;
  case EulerExteriorStateKind::SupersonicOutlet:
    Uplus = Uminus;
    return;
  case EulerExteriorStateKind::SlipWall:
  {
    Uplus = Uminus;
    const double rho = std::fmax(Uminus(0), density_floor);
    Vector nhat(nor);
    const double nrm = nhat.Norml2();
    if (nrm > 0.0)
    {
      nhat /= nrm;
    }
    Vector velocity(dim);
    for (int d = 0; d < dim; ++d)
    {
      velocity(d) = Uminus(1 + d) / rho;
    }
    const double normal_velocity = InnerProduct(velocity, nhat);
    Uplus(0) = rho;
    for (int d = 0; d < dim; ++d)
    {
      Uplus(1 + d) = rho * (velocity(d) - 2.0 * normal_velocity * nhat(d));
    }
    return;
  }
  case EulerExteriorStateKind::SubsonicInlet:
  case EulerExteriorStateKind::SubsonicOutletPressure:
  case EulerExteriorStateKind::FarField:
  case EulerExteriorStateKind::NonReflectingOutlet:
    break;
  }

  mfem::Vector nhat, that;
  BuildNormalTangential(nor, dim, nhat, that);
  const PrimitiveNormalState inner =
    ConservativeToPrimitiveNormal(Uminus,
                                  dim,
                                  gamma,
                                  nhat,
                                  that,
                                  density_floor,
                                  pressure_floor);

  if (kind == EulerExteriorStateKind::SubsonicOutletPressure)
  {
    const PrimitiveNormalState bnd =
      SubsonicOutletPressureClosure(
        inner, gamma, pressure_out, density_floor, pressure_floor);
    PrimitiveNormalToConservative(bnd, dim, gamma, nhat, that, Uplus);
    return;
  }

  MFEM_VERIFY(reference_state != nullptr,
              "Characteristic exterior state requires a reference state.");
  const PrimitiveNormalState reference = ConservativeToPrimitiveNormal(
    *reference_state,
    dim,
    gamma,
    nhat,
    that,
    density_floor,
    pressure_floor);
  if (kind == EulerExteriorStateKind::NonReflectingOutlet)
  {
    const PrimitiveNormalState bnd = NonReflectingOutletClosure(
      inner, reference, gamma, density_floor, pressure_floor);
    PrimitiveNormalToConservative(bnd, dim, gamma, nhat, that, Uplus);
    return;
  }
  if (kind == EulerExteriorStateKind::SubsonicInlet)
  {
    const PrimitiveNormalState bnd = SubsonicInletClosure(
      inner, reference, gamma, density_floor, pressure_floor);
    PrimitiveNormalToConservative(bnd, dim, gamma, nhat, that, Uplus);
    return;
  }

  if (inner.un <= -inner.c)
  {
    Uplus = *reference_state;
    return;
  }
  if (inner.un >= inner.c)
  {
    Uplus = Uminus;
    return;
  }
  const PrimitiveNormalState bnd = inner.un < 0.0
      ? SubsonicInletClosure(
          inner, reference, gamma, density_floor, pressure_floor)
      : NonReflectingOutletClosure(
          inner, reference, gamma, density_floor, pressure_floor);
  PrimitiveNormalToConservative(bnd, dim, gamma, nhat, that, Uplus);
}

// ---------- SupersonicInletState ----------

SupersonicInletState::SupersonicInletState(const Vector& Uin)
  : Uin_(Uin)
{
}

void SupersonicInletState::Eval(const Vector& Uminus,
                                const Vector& nor,
                                FaceElementTransformations&,
                                Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::SupersonicInlet,
                             Uminus, nor, 1, 1.4, &Uin_, 0.0, Uplus);
}

// ---------- SupersonicOutletState ----------

void SupersonicOutletState::Eval(const Vector& Uminus,
                                 const Vector& nor,
                                 FaceElementTransformations&,
                                 Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::SupersonicOutlet,
                             Uminus, nor, 1, 1.4, nullptr, 0.0, Uplus);
}

// ---------- SubsonicOutletPressureState ----------

SubsonicOutletPressureState::SubsonicOutletPressureState(int dim,
                                                         double gamma,
                                                         double pressure_out,
                                                         double density_floor,
                                                         double pressure_floor)
  : dim_(dim),
    gamma_(gamma),
    p_out_(pressure_out),
    density_floor_(density_floor),
    pressure_floor_(pressure_floor)
{
}

void SubsonicOutletPressureState::Eval(const Vector& Uminus,
                                       const Vector& nor,
                                       FaceElementTransformations&,
                                       Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::SubsonicOutletPressure,
                             Uminus,
                             nor,
                             dim_,
                             gamma_,
                             nullptr,
                             p_out_,
                             Uplus,
                             density_floor_,
                             pressure_floor_);
}

// ---------- NonReflectingOutletState ----------

NonReflectingOutletState::NonReflectingOutletState(const Vector& Uref,
                                                   int dim,
                                                   double gamma,
                                                   double density_floor,
                                                   double pressure_floor)
  : Uref_(Uref),
    dim_(dim),
    gamma_(gamma),
    density_floor_(density_floor),
    pressure_floor_(pressure_floor)
{
}

void NonReflectingOutletState::Eval(const Vector& Uminus,
                                    const Vector& nor,
                                    FaceElementTransformations&,
                                    Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::NonReflectingOutlet,
                             Uminus,
                             nor,
                             dim_,
                             gamma_,
                             &Uref_,
                             0.0,
                             Uplus,
                             density_floor_,
                             pressure_floor_);
}

// ---------- SubsonicInletState ----------

SubsonicInletState::SubsonicInletState(const Vector& Uin,
                                       int dim,
                                       double gamma,
                                       double density_floor,
                                       double pressure_floor)
  : Uin_(Uin),
    dim_(dim),
    gamma_(gamma),
    density_floor_(density_floor),
    pressure_floor_(pressure_floor)
{
}

void SubsonicInletState::Eval(const Vector& Uminus,
                              const Vector& nor,
                              FaceElementTransformations&,
                              Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::SubsonicInlet,
                             Uminus,
                             nor,
                             dim_,
                             gamma_,
                             &Uin_,
                             0.0,
                             Uplus,
                             density_floor_,
                             pressure_floor_);
}

// ---------- FarFieldState ----------

FarFieldState::FarFieldState(const Vector& Ufar,
                             int dim,
                             double gamma,
                             double density_floor,
                             double pressure_floor)
  : Ufar_(Ufar),
    dim_(dim),
    gamma_(gamma),
    density_floor_(density_floor),
    pressure_floor_(pressure_floor)
{
}

void FarFieldState::Eval(const Vector& Uminus,
                         const Vector& nor,
                         FaceElementTransformations&,
                         Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::FarField,
                             Uminus,
                             nor,
                             dim_,
                             gamma_,
                             &Ufar_,
                             0.0,
                             Uplus,
                             density_floor_,
                             pressure_floor_);
}

// ---------- SlipWallState ----------

SlipWallState::SlipWallState(int dim,
                             double density_floor,
                             double pressure_floor)
  : dim_(dim),
    density_floor_(density_floor),
    pressure_floor_(pressure_floor)
{
}

void SlipWallState::Eval(const Vector& Uminus,
                         const Vector& nor,
                         FaceElementTransformations&,
                         Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::SlipWall,
                             Uminus,
                             nor,
                             dim_,
                             1.4,
                             nullptr,
                             0.0,
                             Uplus,
                             density_floor_,
                             pressure_floor_);
}

} // namespace mfem

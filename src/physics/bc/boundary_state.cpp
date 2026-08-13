#include "physics/bc/boundary_state.hpp"

#include <algorithm>
#include <cmath>

namespace mfem
{

namespace
{

constexpr double kRhoFloor = 1.0e-12;
constexpr double kPFloor = 1.0e-12;

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
                                                   const mfem::Vector& that)
{
  PrimitiveNormalState out;
  MFEM_VERIFY(U.Size() >= dim + 2,
              "Conservative state has insufficient size for Euler variables.");

  const double rho = std::max(U(0), kRhoFloor);
  mfem::Vector vel(dim);
  double vel2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    vel(d) = U(1 + d) / rho;
    vel2 += vel(d) * vel(d);
  }

  const double E = U(1 + dim) / rho;
  const double kinetic = 0.5 * vel2;
  const double p = std::max((gamma - 1.0) * rho * (E - kinetic), kPFloor);
  const double c = std::sqrt(std::max(gamma * p / rho, 0.0));

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

inline double EntropyLike(double p, double rho, double gamma)
{
  return p / std::pow(std::max(rho, kRhoFloor), gamma);
}

PrimitiveNormalState SubsonicOutletPressureClosure(const PrimitiveNormalState& inner,
                                                   double gamma,
                                                   double p_out)
{
  PrimitiveNormalState out = inner;
  if (inner.un >= inner.c || inner.un <= -inner.c)
  {
    return out; // supersonic or reversed branch -> transmissive fallback
  }

  const double g1 = gamma - 1.0;
  const double s = EntropyLike(inner.p, inner.rho, gamma);
  const double p = std::max(p_out, kPFloor);
  const double rho = std::pow(p / s, 1.0 / gamma);
  const double c = std::sqrt(std::max(gamma * p / rho, 0.0));
  const double j_plus = inner.un + 2.0 * inner.c / g1;
  const double un = j_plus - 2.0 * c / g1;

  out.rho = std::max(rho, kRhoFloor);
  out.p = p;
  out.c = c;
  out.un = un;
  out.ut = inner.ut;
  return out;
}

PrimitiveNormalState NonReflectingOutletClosure(const PrimitiveNormalState& inner,
                                                const PrimitiveNormalState& ref,
                                                double gamma)
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

  const double s_ref = EntropyLike(ref.p, ref.rho, gamma);
  const double rho =
    std::pow((c * c) / std::max(gamma * s_ref, kPFloor), 1.0 / g1);
  const double p = s_ref * std::pow(std::max(rho, kRhoFloor), gamma);

  out.rho = std::max(rho, kRhoFloor);
  out.p = std::max(p, kPFloor);
  out.c = c;
  out.un = un;
  out.ut = inner.ut;
  return out;
}

PrimitiveNormalState SubsonicInletClosure(const PrimitiveNormalState& inner,
                                          const PrimitiveNormalState& target,
                                          double gamma)
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

  const double s_in = EntropyLike(target.p, target.rho, gamma);
  const double rho =
    std::pow((c * c) / std::max(gamma * s_in, kPFloor), 1.0 / g1);
  const double p = s_in * std::pow(std::max(rho, kRhoFloor), gamma);

  out.rho = std::max(rho, kRhoFloor);
  out.p = std::max(p, kPFloor);
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
                                Vector& Uplus)
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
    const double rho = std::max(Uminus(0), kRhoFloor);
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
    ConservativeToPrimitiveNormal(Uminus, dim, gamma, nhat, that);

  if (kind == EulerExteriorStateKind::SubsonicOutletPressure)
  {
    const PrimitiveNormalState bnd =
      SubsonicOutletPressureClosure(inner, gamma, pressure_out);
    PrimitiveNormalToConservative(bnd, dim, gamma, nhat, that, Uplus);
    return;
  }

  MFEM_VERIFY(reference_state != nullptr,
              "Characteristic exterior state requires a reference state.");
  const PrimitiveNormalState reference =
    ConservativeToPrimitiveNormal(*reference_state, dim, gamma, nhat, that);
  if (kind == EulerExteriorStateKind::NonReflectingOutlet)
  {
    const PrimitiveNormalState bnd =
      NonReflectingOutletClosure(inner, reference, gamma);
    PrimitiveNormalToConservative(bnd, dim, gamma, nhat, that, Uplus);
    return;
  }
  if (kind == EulerExteriorStateKind::SubsonicInlet)
  {
    const PrimitiveNormalState bnd =
      SubsonicInletClosure(inner, reference, gamma);
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
      ? SubsonicInletClosure(inner, reference, gamma)
      : NonReflectingOutletClosure(inner, reference, gamma);
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
                                                         double pressure_out)
  : dim_(dim),
    gamma_(gamma),
    p_out_(pressure_out)
{
}

void SubsonicOutletPressureState::Eval(const Vector& Uminus,
                                       const Vector& nor,
                                       FaceElementTransformations&,
                                       Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::SubsonicOutletPressure,
                             Uminus, nor, dim_, gamma_, nullptr, p_out_, Uplus);
}

// ---------- NonReflectingOutletState ----------

NonReflectingOutletState::NonReflectingOutletState(const Vector& Uref,
                                                   int dim,
                                                   double gamma)
  : Uref_(Uref),
    dim_(dim),
    gamma_(gamma)
{
}

void NonReflectingOutletState::Eval(const Vector& Uminus,
                                    const Vector& nor,
                                    FaceElementTransformations&,
                                    Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::NonReflectingOutlet,
                             Uminus, nor, dim_, gamma_, &Uref_, 0.0, Uplus);
}

// ---------- SubsonicInletState ----------

SubsonicInletState::SubsonicInletState(const Vector& Uin, int dim, double gamma)
  : Uin_(Uin),
    dim_(dim),
    gamma_(gamma)
{
}

void SubsonicInletState::Eval(const Vector& Uminus,
                              const Vector& nor,
                              FaceElementTransformations&,
                              Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::SubsonicInlet,
                             Uminus, nor, dim_, gamma_, &Uin_, 0.0, Uplus);
}

// ---------- FarFieldState ----------

FarFieldState::FarFieldState(const Vector& Ufar, int dim, double gamma)
  : Ufar_(Ufar),
    dim_(dim),
    gamma_(gamma)
{
}

void FarFieldState::Eval(const Vector& Uminus,
                         const Vector& nor,
                         FaceElementTransformations&,
                         Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::FarField,
                             Uminus, nor, dim_, gamma_, &Ufar_, 0.0, Uplus);
}

// ---------- SlipWallState ----------

SlipWallState::SlipWallState(int dim)
  : dim_(dim)
{
}

void SlipWallState::Eval(const Vector& Uminus,
                         const Vector& nor,
                         FaceElementTransformations&,
                         Vector& Uplus) const
{
  EvaluateEulerExteriorState(EulerExteriorStateKind::SlipWall,
                             Uminus, nor, dim_, 1.4, nullptr, 0.0, Uplus);
}

} // namespace mfem

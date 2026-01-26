#include "physics/bc/boundary_state.hpp"

namespace mfem
{

// ---------- SupersonicInletState ----------

SupersonicInletState::SupersonicInletState(const Vector& Uin)
  : Uin_(Uin) {}

void SupersonicInletState::Eval(const Vector&,
                                const Vector&,
                                FaceElementTransformations&,
                                Vector& Uplus) const
{
  Uplus = Uin_;
}

// ---------- SupersonicOutletState ----------

void SupersonicOutletState::Eval(const Vector& Uminus,
                                 const Vector&,
                                 FaceElementTransformations&,
                                 Vector& Uplus) const
{
  Uplus = Uminus;
}

// ---------- SlipWallState ----------

SlipWallState::SlipWallState(int dim)
  : dim_(dim) {}

void SlipWallState::Eval(const Vector& Uminus,
                         const Vector& nor,
                         FaceElementTransformations&,
                         Vector& Uplus) const
{
  // Copy conservative state
  Uplus = Uminus;

  const double rho = Uminus(0);

  // Unit normal from scaled normal
  Vector nhat(nor);
  const double nrm = nhat.Norml2();
  if (nrm > 0.0) { nhat /= nrm; }

  // Reconstruct velocity u^- from momentum
  Vector u(dim_);
  for (int d = 0; d < dim_; ++d) { u(d) = Uminus(1 + d) / rho; }

  // Reflect normal component
  const double un = InnerProduct(u, nhat);
  for (int d = 0; d < dim_; ++d)
  {
    const double uprime_d = u(d) - 2.0 * un * nhat(d);
    Uplus(1 + d) = rho * uprime_d; // new momentum
  }
  // rho (index 0) and E (index 1+dim_) remain unchanged (adiabatic slip wall)
}

} // namespace mfem
